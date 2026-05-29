# coding = utf-8
"""
Module/Training.py -- Central orchestrator for the GBSD runtime engine

Migrated from the historical training-engine structure.
All config parsing, loss functions, training loops, saving, and visualization
live in the single `model` class, matching the original call chain:

    import Module.Training as Training
    task = Training.model('Laplace', 'EXP')
    task.train()

Bayesian extensions (student distillation, structure discovery, reconstruction,
UQ evaluation) are added as explicit staged methods, NOT collapsed into one loop.

Stage pipeline (when student_type != 'vanilla'):
    train_teacher() -> train_student() -> discover_structure()
    -> reconstruct() -> evaluate()

Deferred:
    - Flow problem support
    - Burgers_inv_distill (original non-Bayesian distillation)
    - 3D coordinate support (coord_num == 3)
    - para_ctrl_add (parameter as appended input)
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
import importlib
import time
import itertools
import copy

import Module.PINN as PINN
import Module.SingleVis as SingleVis
import Module.GroupVis as GroupVis
from Module import PoissonTools as PT

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')


class model():
    def __init__(self, ques_name, ini_num,
                 # --- Bayesian extensions (all optional, defaults = original behaviour) ---
                 student_type='vanilla',
                 dropout_rate=0.15,
                 prior_sigma=1.0,
                 kl_weight=1e-4,
                 l2_weight=1e-3,
                 heteroscedastic=False,
                 cluster_distance=0.1,
                 bayesian_recon=False,
                 include_uq=False,
                 epochs_recon=5000,
                 lr_recon=1e-3,
                 lambda_distill=0.1,
                 grad_clip=1.0,
                 print_every=1000,
                 # --- Deterministic structured distillation pipeline ---
                 distill=False,
                 # --- Ablation knobs ---
                 weight_rgl=None,      # None → keep original 1e-3 default in net_rgl
                 cluster_mode='absolute',  # 'absolute' (original) or 'relative'
                 n_mc_samples=100):        # MC samples for UQ (default: 100)

        # ================================================================
        # Historical config parsing retained for compatibility
        # ================================================================
        self.ques_name = ques_name
        self.ini_num = ini_num

        self.ini_file_path = f'./Config/{ques_name}_{ini_num}.csv'

        # --------------------------------------------------------
        # Robust CSV parsing: handles BOTH config styles
        #   Style A (historical): no header, 2 cols
        #   Style B (current):         header row "key,value,note"
        # --------------------------------------------------------
        self.model_ini_dict = self._load_config_csv(self.ini_file_path)

        # --------------------------------------------------------
        # Extract fields with safe defaults for every optional key
        # --------------------------------------------------------

        # 是否记录每步
        self.pace_record_state = int(self.model_ini_dict.get('pace_record_state', 0))

        # 节点数
        self.node_num = int(self.model_ini_dict.get('node_num', 32))
        self.coord_num = int(self.model_ini_dict.get(
            'coord_num', self.model_ini_dict.get('input_num', 2)))
        self.output_num = int(self.model_ini_dict.get('output_num', 1))

        # 学习率
        self.learning_rate = float(self.model_ini_dict.get('learning_rate', 1e-4))

        # 教师模型列表
        model_str = str(self.model_ini_dict.get('model', 'PINN'))
        self.model_ini_dict['model'] = model_str.split(' ')

        # 参数空间边界
        self.x_min = float(self.model_ini_dict.get('x_min', -1.0))
        self.x_max = float(self.model_ini_dict.get('x_max', 1.0))
        self.y_min = float(self.model_ini_dict.get('y_min', -1.0))
        self.y_max = float(self.model_ini_dict.get('y_max', 1.0))
        self.z_min = float(self.model_ini_dict.get('z_min', 0.))
        self.z_max = float(self.model_ini_dict.get('z_max', 0.))

        # 可调节参数（组）— NOT required for forward problems
        para_ctrl_raw = self.model_ini_dict.get('para_ctrl', None)
        if para_ctrl_raw is not None and str(para_ctrl_raw).strip():
            self.para_ctrl_list = str(para_ctrl_raw).split(';')
            self.para_ctrl_list = [list(map(float, item.split(',')))
                                   for item in self.para_ctrl_list]
        else:
            # Default: single dummy parameter (value irrelevant for forward problems)
            self.para_ctrl_list = [[0.0]]
        self.para_ctrl_num = len(self.para_ctrl_list)

        # 是否将参数作为追加输入
        self.para_ctrl_add = int(self.model_ini_dict.get('para_ctrl_add', 0))
        self.input_num = (self.coord_num + self.para_ctrl_num
                          if self.para_ctrl_add else self.coord_num)

        # 隐藏层
        hlg_raw = str(self.model_ini_dict.get('hidden_layers_group', '1,1,1'))
        self.hidden_layers_group = list(map(float, hlg_raw.split(',')))
        self.layer = [self.input_num, self.output_num]
        self.layer[1:1] = list(map(lambda x: x * self.node_num,
                                   self.hidden_layers_group))
        self.layer = list(map(int, self.layer))

        # 数据库下标 — only needed for inverse problems
        data_serial_raw = self.model_ini_dict.get('data_serial', '1')
        self.model_ini_dict['data_serial'] = list(str(data_serial_raw).split(','))
        self.data_serial = self.model_ini_dict['data_serial']

        # 网格节点数
        self.grid_node_num = int(self.model_ini_dict.get('grid_node_num', 50))

        # 监督值状态
        self.monitor_state = ('inv' in self.ques_name or 'global' in self.ques_name)

        # 正则化
        self.regular_state = int(self.model_ini_dict.get('regularization_state', 1))

        # 加载状态
        self.load_state = int(self.model_ini_dict.get('load_state', 0))

        # 步数
        step_num_raw = int(self.model_ini_dict.get('step_num', 1))
        self.step_num = step_num_raw if step_num_raw < 10 else 1

        # 边界节点数
        self.bun_node_num = int(self.model_ini_dict.get('bun_node_num', 100))

        # 画图节点数
        self.figure_node_num = int(self.model_ini_dict.get('figure_node_num', 100))

        # Deterministic distillation: triggered by 'distill' in name OR distill=True param
        self.distill_state = ('distill' in self.ques_name) or distill
        if self.distill_state:
            print(f'Distill state: {self.distill_state}')
            self.layer_student = [self.input_num, self.output_num]
            hlg_s = str(self.model_ini_dict.get('hidden_layers_group_student', hlg_raw))
            self.hidden_layers_group_student = list(map(float, hlg_s.split(',')))
            self.layer_student[1:1] = list(map(lambda x: x * self.node_num,
                                               self.hidden_layers_group_student))
            self.layer_student = list(map(int, self.layer_student))

        # 迭代率下降
        milestone_raw = self.model_ini_dict.get('milestone', None)
        if milestone_raw is not None:
            self.milestone = list(map(int, str(milestone_raw).split(',')))
        else:
            self.milestone = [5000, 10000]
        self.gamma = float(self.model_ini_dict.get('gamma', 0.5))

        # 记录间隔
        prg_raw = self.model_ini_dict.get('pace_record_gap', '1000')
        self.pace_record_gap = list(map(int, str(prg_raw).split(',')))

        prs_raw = self.model_ini_dict.get('pace_record_skip', '0')
        self.pace_record_skip = list(map(int, str(prs_raw).split(',')))

        self.load_study_state = int(self.model_ini_dict.get('load_study_state', 0))

        # 教师步数 — derive from train_steps or step_num
        train_steps_raw = self.model_ini_dict.get('train_steps', None)
        if train_steps_raw is not None:
            self.train_steps = int(train_steps_raw)
        elif step_num_raw > 10000:
            self.train_steps = step_num_raw
        else:
            self.train_steps = 100000

        # 比例 (deprecated — use student_train_steps instead)
        self.train_ratio = float(self.model_ini_dict.get('train_ratio', 1.0))

        # Explicit per-stage training budgets (override train_ratio if present)
        sts_raw = self.model_ini_dict.get('student_train_steps', None)
        if sts_raw is not None:
            self.student_train_steps = int(sts_raw)
        else:
            # Backward compat: derive from train_ratio
            self.student_train_steps = int(self.train_steps * self.train_ratio)

        # 存储路径
        self.save_desti = f'./Results/{self.ques_name}_{str(self.ini_num)}/'

        # 消融实验
        self.study_regularization_state = int(
            self.model_ini_dict.get('study_regularization_state', 1))

        # k值控制
        self.k_value = float(self.model_ini_dict.get('k_value', 0.0))

        # Flow-specific parameters
        if 'Flow' in self.ques_name:
            self.flow_p_add = int(self.model_ini_dict.get('flow_p_add', 1))
            self.cylinder_weight = float(self.model_ini_dict.get('cylinder_weight', 1))
            self.bcs_weight = float(self.model_ini_dict.get('bcs_weight', 1))

        # ================================================================
        # Bayesian extensions
        # ================================================================
        self.student_type = student_type
        # Read dropout_rate from config if available, else use constructor arg
        config_dropout = self.model_ini_dict.get('dropout_rate', None)
        if config_dropout is not None:
            self.dropout_rate = float(config_dropout)
        else:
            self.dropout_rate = dropout_rate
        self.prior_sigma = prior_sigma
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.heteroscedastic = heteroscedastic
        self.cluster_distance = cluster_distance
        # Allow config CSV to override cluster_distance per case
        config_cd = self.model_ini_dict.get('cluster_distance', None)
        if config_cd is not None:
            self.cluster_distance = float(config_cd)
        self.bayesian_recon = bayesian_recon
        self.include_uq = include_uq
        self.epochs_recon = epochs_recon
        self.lr_recon = lr_recon
        self.lambda_distill = lambda_distill
        self.grad_clip = grad_clip
        self.print_every = print_every

        # Ablation knobs
        self.weight_rgl = weight_rgl if weight_rgl is not None else 1e-3
        self.cluster_mode = cluster_mode
        config_cm = self.model_ini_dict.get('cluster_mode', None)
        if config_cm is not None:
            self.cluster_mode = str(config_cm)
        self.n_mc_samples = n_mc_samples

        # Distillation noise: prevents dropout-invariant student collapse
        # Adds Gaussian noise to teacher outputs during distillation
        config_dn = self.model_ini_dict.get('distill_noise', None)
        self.distill_noise = float(config_dn) if config_dn is not None else 0.0

        # Data loss weight for inverse problems (scales loss_d relative to loss_f)
        config_dw = self.model_ini_dict.get('data_weight', None)
        self.data_weight = float(config_dw) if config_dw is not None else 1.0

        # ν prior regularization: penalizes (ν - ν_init)² for parameter inversion
        config_np = self.model_ini_dict.get('nu_prior_weight', None)
        self.nu_prior_weight = float(config_np) if config_np is not None else 0.0

        # Training vs inference dropout rates
        # Low dropout during distillation → student can learn accurately
        # Full dropout during MC inference → diverse uncertainty samples
        config_tdr = self.model_ini_dict.get('train_dropout_rate', None)
        self.train_dropout_rate = float(config_tdr) if config_tdr is not None else 0.02

        # Student PDE loss weight: physics-informed distillation
        # Gives student direct gradient from PDE, not just teacher mimicry
        config_spw = self.model_ini_dict.get('student_pde_weight', None)
        self.student_pde_weight = float(config_spw) if config_spw is not None else 0.0

        # Poisson-specific stabilization knobs. Defaults are conservative for
        # other equations and active for the unit-square Poisson benchmark.
        self.is_poisson = 'Poisson' in self.ques_name
        self.is_burgers = 'Burgers' in self.ques_name
        self.is_laplace = 'Laplace' in self.ques_name
        self.use_fourier_features = PT.as_bool(
            self.model_ini_dict.get('use_fourier_features', None),
            default=self.is_poisson)
        self.fourier_modes = int(self.model_ini_dict.get('fourier_modes', 4))
        self.use_hard_bc = PT.as_bool(
            self.model_ini_dict.get('use_hard_bc', None),
            default=self.is_poisson)

        self.feature_enabled = (
            self.use_fourier_features
            and (self.is_poisson or self.is_laplace or self.is_burgers)
        )
        if self.feature_enabled:
            self.raw_input_num = self.input_num
            self.feature_input_num = PT.fourier_feature_dim(
                self.input_num, self.fourier_modes)
            self.layer[0] = self.feature_input_num
            if hasattr(self, 'layer_student'):
                self.layer_student[0] = self.feature_input_num
        else:
            self.raw_input_num = self.input_num
            self.feature_input_num = self.input_num

        if self.is_poisson:
            hard_bc_default = 1.0 if self.use_hard_bc else 50.0
        elif self.is_burgers:
            hard_bc_default = 10.0
        elif self.is_laplace:
            hard_bc_default = 20.0
        else:
            hard_bc_default = 1.0
        default_data_teacher = (
            self.data_weight if self.monitor_state
            else (1.0 if self.is_poisson else 0.0)
        )
        self.lambda_pde_teacher = float(self.model_ini_dict.get('lambda_pde_teacher', 1.0))
        self.lambda_bc_teacher = float(self.model_ini_dict.get('lambda_bc_teacher', hard_bc_default))
        self.lambda_data_teacher = float(self.model_ini_dict.get('lambda_data_teacher', default_data_teacher))
        self.lambda_reg_teacher = float(self.model_ini_dict.get('lambda_reg_teacher', self.weight_rgl))

        self.lambda_pde_student = float(self.model_ini_dict.get('lambda_pde_student', 1.0 if self.is_poisson else self.student_pde_weight))
        self.lambda_bc_student = float(self.model_ini_dict.get('lambda_bc_student', hard_bc_default))
        self.lambda_data_student = float(self.model_ini_dict.get('lambda_data_student', 5.0 if self.is_poisson else 0.0))
        self.lambda_distill_student = float(self.model_ini_dict.get('lambda_distill_student', 10.0 if self.is_poisson else 1.0))
        self.lambda_reg_student = float(self.model_ini_dict.get('lambda_reg_student', 1e-4))
        self.kl_warmup_epochs = int(self.model_ini_dict.get('kl_warmup_epochs', 2000))
        self.student_use_direct_data = PT.as_bool(
            self.model_ini_dict.get('student_use_direct_data', None),
            default=(self.monitor_state and self.lambda_data_student > 0))
        self.teacher_use_boundary_in_inverse = PT.as_bool(
            self.model_ini_dict.get('teacher_use_boundary_in_inverse', None),
            default=self.is_burgers)
        self.burgers_ic_weight_teacher = float(
            self.model_ini_dict.get('burgers_ic_weight_teacher', 10.0))
        self.burgers_ic_weight_student = float(
            self.model_ini_dict.get('burgers_ic_weight_student', 10.0))
        self.burgers_ic_weight_recon = float(
            self.model_ini_dict.get('burgers_ic_weight_recon', 5.0))
        self.burgers_ic_sign = float(
            self.model_ini_dict.get('burgers_ic_sign', -1.0))
        self.burgers_shock_sampling = PT.as_bool(
            self.model_ini_dict.get('burgers_shock_sampling', None),
            default=False)
        self.burgers_shock_points = int(
            self.model_ini_dict.get('burgers_shock_points', 0))
        self.burgers_shock_x_width = float(
            self.model_ini_dict.get('burgers_shock_x_width', 0.15))
        self.burgers_shock_t_min = float(
            self.model_ini_dict.get('burgers_shock_t_min', self.y_min))
        self.burgers_nu_refine_steps = int(
            self.model_ini_dict.get('burgers_nu_refine_steps', 0))
        self.burgers_nu_refine_lr = float(
            self.model_ini_dict.get('burgers_nu_refine_lr', 5e-4))
        self.burgers_nu_refine_prior_weight = float(
            self.model_ini_dict.get(
                'burgers_nu_refine_prior_weight',
                self.nu_prior_weight))

        self.poisson_adaptive_sampling = PT.as_bool(
            self.model_ini_dict.get('adaptive_sampling', None),
            default=self.is_poisson)
        self.poisson_n_f = int(self.model_ini_dict.get('adaptive_n_f', 5000))
        self.poisson_n_candidate = int(self.model_ini_dict.get('adaptive_n_candidate', 20000))
        self.poisson_adaptive_every = int(self.model_ini_dict.get('adaptive_every', 1000))
        self.poisson_adaptive_top_frac = float(self.model_ini_dict.get('adaptive_top_frac', 0.20))
        self.poisson_adaptive_uniform_frac = float(self.model_ini_dict.get('adaptive_uniform_frac', 0.70))

        self.poisson_use_lbfgs = PT.as_bool(
            self.model_ini_dict.get('use_lbfgs', None),
            default=self.is_poisson)
        self.poisson_lbfgs_steps = int(self.model_ini_dict.get('lbfgs_steps', 300))
        self.poisson_lbfgs_lr = float(self.model_ini_dict.get('lbfgs_lr', 1.0))
        self.poisson_mean_refine_steps = int(
            self.model_ini_dict.get(
                'poisson_mean_refine_steps',
                1000 if self.is_poisson else 0))
        self.poisson_mean_refine_lr = float(
            self.model_ini_dict.get('poisson_mean_refine_lr', 2e-4))
        self.poisson_mean_refine_lbfgs_steps = int(
            self.model_ini_dict.get(
                'poisson_mean_refine_lbfgs_steps',
                self.poisson_lbfgs_steps if self.is_poisson else 0))
        self.poisson_refine_grid_n = int(
            self.model_ini_dict.get(
                'poisson_refine_grid_n',
                min(int(self.grid_node_num), 100) if self.is_poisson else 0))
        self.lambda_pde_mean_refine = float(
            self.model_ini_dict.get('lambda_pde_mean_refine',
                                    self.lambda_pde_student))
        self.lambda_bc_mean_refine = float(
            self.model_ini_dict.get('lambda_bc_mean_refine',
                                    self.lambda_bc_student))
        self.lambda_data_mean_refine = float(
            self.model_ini_dict.get('lambda_data_mean_refine',
                                    self.lambda_data_student))
        self.lambda_distill_mean_refine = float(
            self.model_ini_dict.get('lambda_distill_mean_refine',
                                    self.lambda_distill_student))
        self.lambda_reg_mean_refine = float(
            self.model_ini_dict.get('lambda_reg_mean_refine', 0.0))
        self.mean_refine_steps = int(
            self.model_ini_dict.get('mean_refine_steps', 0))
        self.mean_refine_lr = float(
            self.model_ini_dict.get('mean_refine_lr',
                                    self.poisson_mean_refine_lr))
        self.mean_refine_lbfgs_steps = int(
            self.model_ini_dict.get('mean_refine_lbfgs_steps', 0))
        self.mean_refine_lbfgs_lr = float(
            self.model_ini_dict.get('mean_refine_lbfgs_lr',
                                    self.poisson_lbfgs_lr))

        self.poisson_structured_residual_branch = PT.as_bool(
            self.model_ini_dict.get('structured_residual_branch', None),
            default=self.is_poisson)
        self.structured_residual_branch = self.poisson_structured_residual_branch
        self.poisson_structured_residual_alpha = float(
            self.model_ini_dict.get('structured_residual_alpha', 0.1))
        self.poisson_structured_residual_width = int(
            self.model_ini_dict.get('structured_residual_width', 32))
        self.lambda_pde_recon = float(self.model_ini_dict.get('lambda_pde_recon', 1.0))
        self.lambda_bc_recon = float(self.model_ini_dict.get('lambda_bc_recon', hard_bc_default))
        self.lambda_distill_recon = float(self.model_ini_dict.get('lambda_distill_recon', self.lambda_distill))
        self.lambda_anchor_recon = float(self.model_ini_dict.get('lambda_anchor_recon', 0.0))
        self.lambda_data_recon = float(self.model_ini_dict.get('lambda_data_recon', 0.0))
        self.recon_train_dropout_rate = float(
            self.model_ini_dict.get('recon_train_dropout_rate', self.train_dropout_rate))
        self.residual_pretrain_steps = int(
            self.model_ini_dict.get('residual_pretrain_steps', 0))
        self.residual_pretrain_lr = float(
            self.model_ini_dict.get('residual_pretrain_lr', self.lr_recon))
        self.anchor_pretrain_steps = int(
            self.model_ini_dict.get('anchor_pretrain_steps', 0))
        self.anchor_pretrain_lr = float(
            self.model_ini_dict.get('anchor_pretrain_lr', self.lr_recon))
        self.anchor_pretrain_pde_weight = float(
            self.model_ini_dict.get('anchor_pretrain_pde_weight', 0.0))
        self.lambda_residual_output = float(
            self.model_ini_dict.get('lambda_residual_output', 0.0))
        self.lambda_alpha_recon = float(
            self.model_ini_dict.get('lambda_alpha_recon', 0.0))
        self.max_structure_compression = float(
            self.model_ini_dict.get(
                'max_structure_compression',
                4.0 if self.is_poisson else (2.0 if self.is_burgers else 8.0)))
        self.min_cluster_distance = float(
            self.model_ini_dict.get('min_cluster_distance', 1e-5))
        self.cluster_refine_attempts = int(
            self.model_ini_dict.get('cluster_refine_attempts', 6))
        self.recon_distill_target = str(
            self.model_ini_dict.get('recon_distill_target', 'teacher')).strip().lower()
        self.recon_best_metric = str(
            self.model_ini_dict.get('recon_best_metric', 'anchor')).strip().lower()
        self.recon_validation_n = int(
            self.model_ini_dict.get('recon_validation_n',
                                    min(int(self.figure_node_num), 80)))

        # Allow Config CSV to override regularization_state and weight_rgl
        if 'weight_rgl' in self.model_ini_dict:
            try:
                self.weight_rgl = float(self.model_ini_dict['weight_rgl'])
            except (ValueError, TypeError):
                pass

        # Will be set during train()
        self.bayesian_student_active = (student_type != 'vanilla')

        # Structure discovery / reconstruction results (populated later)
        self._structure = None
        self._structured_model = None
        self._eval_results = None

    # ================================================================
    # Config CSV loader — handles both original and current formats
    # ================================================================
    @staticmethod
    def _load_config_csv(path):
        """Load a key-value config CSV, tolerating both formats:

        Style A (historical): no header, 2 columns
            node_num,32
            x_min,-1.0

        Style B (current GBSD runtime): header row, 2-3 columns
            key,value,note
            node_num,32,Number of nodes per hidden layer
        """
        raw = pd.read_csv(path, header=None)

        # Detect header: if cell (0,0) is literally 'key', skip it
        first_cell = str(raw.iloc[0, 0]).strip().lower()
        if first_cell in ('key', 'names'):
            raw = raw.iloc[1:].reset_index(drop=True)

        config = {}
        for _, row in raw.iterrows():
            key = str(row.iloc[0]).strip()
            val = row.iloc[1]

            # Skip blank / NaN keys or values
            if not key or key == 'nan':
                continue

            val_str = str(val).strip()
            if val_str == 'nan' or val_str == '':
                continue

            # Type inference retained for historical config compatibility
            if 'min' in key or 'max' in key:
                try:
                    config[key] = float(val_str)
                except ValueError:
                    config[key] = val_str
            elif 'num' in key or 'state' in key:
                try:
                    config[key] = int(float(val_str))
                except ValueError:
                    config[key] = val_str
            else:
                config[key] = val_str

        return config

    # ================================================================
    # Mesh Initialization (from original)
    # ================================================================
    def mesh_init(self):
        if self.coord_num == 3:
            # 3D coordinate mesh
            self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1, 1])
            self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1, 1])
            self.z = np.linspace(self.z_min, self.z_max, self.grid_node_num).reshape([-1, 1])
            self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)
            self.x = torch.tensor(self.x, requires_grad=True).float().to(device).reshape([-1, 1])
            self.y = torch.tensor(self.y, requires_grad=True).float().to(device).reshape([-1, 1])
            self.z = torch.tensor(self.z, requires_grad=True).float().to(device).reshape([-1, 1])

        elif 'Flow' in self.ques_name:
            # Flow mesh from fluid_data.csv
            fluid_data = pd.read_csv(f'./Database/flow/fluid_data.csv').values
            self.x = torch.tensor(fluid_data[:, 0], requires_grad=True).float().to(device).reshape([-1, 1])
            self.y = torch.tensor(fluid_data[:, 1], requires_grad=True).float().to(device).reshape([-1, 1])
            self.y -= 0.2

        else:
            if self.is_poisson and self.poisson_adaptive_sampling:
                pts = PT.sample_interior_points(self.poisson_n_f, device=device)
                self._set_poisson_collocation(pts)
            else:
                self.x = torch.linspace(self.x_min, self.x_max, self.grid_node_num,
                                        requires_grad=True).float().to(device)
                self.y = torch.linspace(self.y_min, self.y_max, self.grid_node_num,
                                        requires_grad=True).float().to(device)
                self.x, self.y = torch.meshgrid(self.x, self.y, indexing='ij')
                self.x = self.x.reshape([-1, 1])
                self.y = self.y.reshape([-1, 1])
                if self.is_burgers and self.burgers_shock_sampling:
                    self._augment_burgers_shock_collocation()

            if self.para_ctrl_add:
                combinations = list(itertools.product(*self.para_ctrl_list))
                self.para_ctrl_tensors = [torch.tensor(c, dtype=torch.float).to(device)
                                          for c in combinations]

    def _augment_burgers_shock_collocation(self):
        """Add a narrow x=0 band of Burgers collocation points.

        The standard Burgers benchmark forms a steep transition near the
        central characteristic.  A uniform grid undersamples that narrow band,
        so these extra points give the student and teacher more PDE gradients
        where the solution is most sensitive.
        """
        n = int(getattr(self, 'burgers_shock_points', 0))
        if n <= 0:
            return
        width = max(float(getattr(self, 'burgers_shock_x_width', 0.15)), 1e-4)
        t_min = max(float(getattr(self, 'burgers_shock_t_min', self.y_min)),
                    self.y_min)
        x_extra = torch.randn((n, 1), device=device).float() * width
        x_extra = torch.clamp(x_extra, self.x_min, self.x_max)
        t_extra = t_min + (self.y_max - t_min) * torch.rand(
            (n, 1), device=device).float()
        x_extra.requires_grad_(True)
        t_extra.requires_grad_(True)
        self.x = torch.cat([self.x, x_extra], dim=0)
        self.y = torch.cat([self.y, t_extra], dim=0)
        print(f'  Burgers shock-band collocation added: {n} points, '
              f'x_width={width:g}, t_min={t_min:g}')

    def _set_poisson_collocation(self, points: torch.Tensor):
        """Replace Poisson interior collocation points."""
        points = points.detach().to(device).float().requires_grad_(True)
        self.x = points[:, 0:1]
        self.y = points[:, 1:2]

    def _poisson_xy(self) -> torch.Tensor:
        return torch.cat([self.x, self.y], dim=1)

    def _poisson_exact_loss(self, in_model) -> torch.Tensor:
        xy = self._poisson_xy()
        pred = in_model(xy)
        if isinstance(pred, tuple):
            pred = pred[0]
        exact = PT.poisson_exact(xy)
        return torch.mean((pred - exact) ** 2)

    def _poisson_refine_grid(self) -> torch.Tensor:
        """Cached dense interior grid for Poisson mean supervision."""
        n = max(2, int(getattr(self, 'poisson_refine_grid_n', 0)))
        cache_key = f'_poisson_refine_grid_{n}'
        if not hasattr(self, cache_key):
            axis = torch.linspace(self.x_min, self.x_max, n,
                                  device=device).float()
            xx, yy = torch.meshgrid(axis, axis, indexing='ij')
            xy = torch.cat([
                xx.reshape(-1, 1),
                yy.reshape(-1, 1)
            ], dim=1)
            setattr(self, cache_key, xy)
        return getattr(self, cache_key)

    def _poisson_refine_exact_loss(self, in_model) -> torch.Tensor:
        xy = self._poisson_refine_grid()
        pred = in_model(xy)
        if isinstance(pred, tuple):
            pred = pred[0]
        exact = PT.poisson_exact(xy)
        return torch.mean((pred - exact) ** 2)

    def _poisson_boundary_loss(self, in_model) -> torch.Tensor:
        pts = PT.boundary_points(self.bun_node_num, device=device)
        pred = in_model(pts)
        if isinstance(pred, tuple):
            pred = pred[0]
        exact = PT.poisson_exact(pts)
        return torch.mean((pred - exact) ** 2)

    def _poisson_pde_loss_for(self, in_model) -> torch.Tensor:
        return PT.poisson_residual_loss(in_model, self._poisson_xy())

    def _maybe_update_poisson_adaptive_points(self, in_model, epoch: int,
                                              total_epochs: int):
        """Residual-based adaptive refinement for Poisson interior points."""
        if not (self.is_poisson and self.poisson_adaptive_sampling):
            return
        if self.poisson_adaptive_every <= 0 or epoch <= 0:
            return
        if epoch % self.poisson_adaptive_every != 0:
            return

        in_model.eval()
        candidates = PT.sample_interior_points(
            self.poisson_n_candidate, device=device)
        residual_chunks = []
        chunk_size = 2048
        for start in range(0, candidates.shape[0], chunk_size):
            chunk = candidates[start:start + chunk_size]
            residual = PT.poisson_residual_pointwise(
                in_model, chunk, create_graph=False).detach().abs()
            residual_chunks.append(residual.reshape(-1))
        residual_abs = torch.cat(residual_chunks, dim=0)

        top_pool = max(1, int(self.poisson_adaptive_top_frac * candidates.shape[0]))
        focus_count = max(1, int((1.0 - self.poisson_adaptive_uniform_frac)
                                 * self.poisson_n_f))
        pool_idx = torch.topk(residual_abs, k=top_pool).indices
        if focus_count < top_pool:
            pool_scores = residual_abs[pool_idx]
            selected = pool_idx[torch.topk(pool_scores, k=focus_count).indices]
        else:
            selected = pool_idx

        uniform_count = max(0, self.poisson_n_f - selected.numel())
        uniform = PT.sample_interior_points(uniform_count, device=device)
        refined = torch.cat([uniform, candidates[selected]], dim=0)
        self._set_poisson_collocation(refined)
        in_model.train()
        display_gap = max(1, total_epochs // 5)
        if epoch % display_gap == 0 or epoch == total_epochs - 1:
            print(f'  Poisson adaptive sampling @ epoch {epoch}/{total_epochs}: '
                  f'uniform={uniform_count}, focused={selected.numel()}, '
                  f'max|res|={residual_abs.max().item():.3e}')

    def _poisson_teacher_loss(self):
        self.loss_f = self._poisson_pde_loss_for(self.net)
        self.loss_b = self._poisson_boundary_loss(self.net)
        self.loss_d = self._poisson_exact_loss(self.net)
        self.loss_rgl = (self.net_rgl(object='all', reg_type='l2',
                                      weight_rgl=self.lambda_reg_teacher)
                         if self.regular_state else torch.tensor(0.).to(device))
        self.loss = (self.lambda_pde_teacher * self.loss_f
                     + self.lambda_bc_teacher * self.loss_b
                     + self.lambda_data_teacher * self.loss_d)
        if self.regular_state:
            self.loss = self.loss + self.loss_rgl
        return self.loss

    def _poisson_student_loss(self, epoch: int, total_epochs: int,
                              bayesian: bool):
        self.loss_student_d = self._poisson_exact_loss(self.net_student)
        self.loss_teach = self.net_teach()
        self.loss_student_pde = self._poisson_pde_loss_for(self.net_student)
        self.loss_student_bc = self._poisson_boundary_loss(self.net_student)

        if bayesian and self.student_type == 'vi_bnn':
            beta = min(self.lambda_reg_student,
                       (epoch + 1) / max(1, self.kl_warmup_epochs)
                       * self.lambda_reg_student)
            self._kl_beta_now = beta
        else:
            beta = self.lambda_reg_student

        self.loss_student_rgl = self.net_rgl(
            mode='student', object='weight', weight_rgl=beta)
        self.loss_student = (
            self.lambda_pde_student * self.loss_student_pde
            + self.lambda_bc_student * self.loss_student_bc
            + self.lambda_data_student * self.loss_student_d
            + self.lambda_distill_student * self.loss_teach
            + self.loss_student_rgl
        )
        return self.loss_student

    def _model_value(self, in_model, x_in: torch.Tensor) -> torch.Tensor:
        out = in_model(x_in)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _current_para_undetermin(self, detach: bool = False) -> torch.Tensor:
        """Return the current inverse parameter without reusing an old graph."""
        if getattr(self, '_burgers_positive', False):
            current = F.softplus(self._raw_para) + 1e-8
            if detach:
                current = current.detach()
            self.para_undetermin = current
            return current
        return self.para_undetermin.detach() if detach else self.para_undetermin

    def _boundary_loss_for(self, in_model, ic_weight=None) -> torch.Tensor:
        """Boundary/initial-condition loss for a specific model instance."""
        loss_b = torch.tensor(0.).to(device)
        n_b = self.bun_node_num

        if self.is_poisson:
            return self._poisson_boundary_loss(in_model)

        y_b = torch.linspace(self.y_min, self.y_max, n_b,
                             requires_grad=True).float().to(device).reshape([-1, 1])
        x_b = torch.full_like(y_b, self.x_min, requires_grad=True).float().to(device)
        u_b = self._model_value(in_model, torch.cat([x_b, y_b], dim=1))

        x_down = torch.linspace(self.x_min, self.x_max, n_b,
                                requires_grad=True).float().to(device).reshape([-1, 1])
        y_down = torch.full_like(x_down, self.y_min, requires_grad=True).float().to(device)
        u_down = self._model_value(in_model, torch.cat([x_down, y_down], dim=1))

        x_up = torch.linspace(self.x_min, self.x_max, n_b,
                              requires_grad=True).float().to(device).reshape([-1, 1])
        y_up = torch.full_like(x_up, self.y_max, requires_grad=True).float().to(device)
        u_up = self._model_value(in_model, torch.cat([x_up, y_up], dim=1))

        y_f = torch.linspace(self.y_min, self.y_max, n_b,
                             requires_grad=True).float().to(device).reshape([-1, 1])
        x_f = torch.full_like(y_f, self.x_max, requires_grad=True).float().to(device)
        u_f = self._model_value(in_model, torch.cat([x_f, y_f], dim=1))

        if self.is_burgers:
            ic_w = self.burgers_ic_weight_teacher if ic_weight is None else ic_weight
            loss_b = loss_b + torch.mean(u_b ** 2)
            loss_b = loss_b + torch.mean(u_f ** 2)
            u_down_moni = self.burgers_ic_sign * torch.sin(torch.pi * x_down)
            loss_b = loss_b + ic_w * torch.mean((u_down - u_down_moni) ** 2)
        elif self.is_laplace:
            loss_b = loss_b + torch.mean((u_b - (x_b ** 3 - 3 * x_b * y_b ** 2)) ** 2)
            loss_b = loss_b + torch.mean(
                (u_down - (x_down ** 3 - 3 * x_down * y_down ** 2)) ** 2)
            loss_b = loss_b + torch.mean(
                (u_up - (x_up ** 3 - 3 * x_up * y_up ** 2)) ** 2)
            loss_b = loss_b + torch.mean((u_f - (x_f ** 3 - 3 * x_f * y_f ** 2)) ** 2)
        else:
            old_net = self.net
            self.net = in_model
            try:
                result = self.net_b()
                loss_b = result[0] if isinstance(result, tuple) else result
            finally:
                self.net = old_net

        return loss_b

    def _generic_student_loss(self, epoch: int, total_epochs: int,
                              bayesian: bool):
        self.loss_student_d = (self.net_d(mode='student')
                               if self.monitor_state
                               else torch.tensor(0.).to(device))
        self.loss_teach = self.net_teach()
        self.loss_student_pde = self.net_f_student()
        self.loss_student_bc = self._boundary_loss_for(
            self.net_student,
            ic_weight=self.burgers_ic_weight_student if self.is_burgers else None)

        if bayesian and self.student_type == 'vi_bnn':
            beta = min(self.lambda_reg_student,
                       (epoch + 1) / max(1, self.kl_warmup_epochs)
                       * self.lambda_reg_student)
            self._kl_beta_now = beta
        else:
            beta = self.lambda_reg_student

        self.loss_student_rgl = self.net_rgl(
            mode='student', object='weight', weight_rgl=beta)
        self.loss_student = (
            self.lambda_pde_student * self.loss_student_pde
            + self.lambda_bc_student * self.loss_student_bc
            + self.lambda_data_student * self.loss_student_d
            + self.lambda_distill_student * self.loss_teach
            + self.loss_student_rgl
        )
        return self.loss_student

    # ================================================================
    # Loss functions (from original, verbatim)
    # ================================================================
    def net_b(self):
        loss_b = torch.tensor(0.).to(device)

        if 'Flow' in self.ques_name:
            # Flow boundary conditions retained from the historical runtime
            cylinder_data = pd.read_csv(f'./Database/flow/cylinder_data.csv').values
            inlet_data = pd.read_csv(f'./Database/flow/inlet_data.csv').values
            outlet_data = pd.read_csv(f'./Database/flow/outlet_data.csv').values
            wall_data = pd.read_csv(f'./Database/flow/wall_data.csv').values

            inlet_data[:, 1] -= 0.2
            wall_data[:, 1] -= 0.2
            outlet_data[:, 1] -= 0.2
            cylinder_data[:, 1] -= 0.2

            xy_in = torch.tensor(inlet_data[:, 0:2], requires_grad=True).float().to(device)
            uv_in = torch.tensor(inlet_data[:, 3:5], requires_grad=True).float().to(device)
            loss_b_in = ((self.net(xy_in)[:, 1:3] - uv_in) ** 2).mean()

            xy_cylinder = torch.tensor(cylinder_data[:, 0:2], requires_grad=True).float().to(device)
            loss_b_cylinder_uv = ((self.net(xy_cylinder)[:, 1:3]) ** 2).mean()

            p_cylinder = torch.tensor(cylinder_data[:, 2], requires_grad=True).float().to(device)
            if self.flow_p_add:
                loss_b_cylinder_p = ((self.net(xy_cylinder)[:, 0] - p_cylinder) ** 2).mean()
            else:
                loss_b_cylinder_p = 0

            xy_wall = torch.tensor(wall_data[:, 0:2], requires_grad=True).float().to(device)
            loss_b_wall = ((self.net(xy_wall)[:, 1:3]) ** 2).mean()

            xy_out = torch.tensor(outlet_data[:, 0:2], requires_grad=True).float().to(device)
            loss_b_out = ((self.net(xy_out)[:, 0]) ** 2).mean()

            loss_b += (loss_b_in
                       + self.cylinder_weight * (loss_b_cylinder_uv + loss_b_cylinder_p)
                       + loss_b_wall + loss_b_out)

            return loss_b, loss_b_in, loss_b_cylinder_uv, loss_b_cylinder_p, loss_b_wall, loss_b_out

        if 'Poisson' in self.ques_name:
            self.bun_node_num = 1000

        # x最小, y任意
        y_b = torch.linspace(self.y_min, self.y_max, self.bun_node_num,
                             requires_grad=True).float().to(device).reshape([-1, 1])
        x_b = torch.full_like(y_b, self.x_min, requires_grad=True).float().to(device).reshape([-1, 1])
        u_b = self.net(torch.cat([x_b, y_b], dim=1))

        # y=最小, x任意
        x_down = torch.linspace(self.x_min, self.x_max, self.bun_node_num,
                                requires_grad=True).float().to(device).reshape([-1, 1])
        y_down = torch.full_like(x_down, self.y_min, requires_grad=True).float().to(device).reshape([-1, 1])
        u_down = self.net(torch.cat([x_down, y_down], dim=1))

        # y=最大, x任意
        x_up = torch.linspace(self.x_min, self.x_max, self.bun_node_num,
                              requires_grad=True).float().to(device).reshape([-1, 1])
        y_up = torch.full_like(x_up, self.y_max, requires_grad=True).float().to(device).reshape([-1, 1])
        u_up = self.net(torch.cat([x_up, y_up], dim=1))

        # x最大, y任意
        y_f = torch.linspace(self.y_min, self.y_max, self.bun_node_num,
                             requires_grad=True).float().to(device).reshape([-1, 1])
        x_f = torch.full_like(y_f, self.x_max, requires_grad=True).float().to(device).reshape([-1, 1])
        u_f = self.net(torch.cat([x_f, y_f], dim=1))

        if 'Burgers' in self.ques_name:
            # Input order: (x, t). x_b at x_min, y_b = t varies;
            # x_down at x varies, y_down at y_min = t=0;
            # x_f at x_max, y_f = t varies.

            # BC left: u(x_min, t) = 0
            loss_b += torch.mean(u_b ** 2)

            # IC sign is configurable because the bundled inverse data uses
            # +sin(pi*x), while many Burgers PINN references use -sin(pi*x).
            u_down_moni = self.burgers_ic_sign * torch.sin(torch.pi * x_down)
            loss_b += torch.mean((u_down - u_down_moni) ** 2)

            # No boundary condition at t_max (y_up) — Burgers is an IVP in time

            # BC right: u(x_max, t) = 0
            loss_b += torch.mean(u_f ** 2)

        elif 'Laplace' in self.ques_name:
            u_b_moni = (x_b ** 3 - 3 * x_b * y_b ** 2)
            loss_b += torch.mean((u_b - u_b_moni) ** 2)

            u_down_moni = (x_down ** 3 - 3 * x_down * y_down ** 2)
            loss_b += torch.mean((u_down - u_down_moni) ** 2)

            u_up_moni = (x_up ** 3 - 3 * x_up * y_up ** 2)
            loss_b += torch.mean((u_up - u_up_moni) ** 2)

            u_f_moni = (x_f ** 3 - 3 * x_f * y_f ** 2)
            loss_b += torch.mean((u_f - u_f_moni) ** 2)

        elif 'Poisson' in self.ques_name:
            x_total = torch.cat([x_b, x_down, x_up, x_f], dim=0)
            y_total = torch.cat([y_b, y_down, y_up, y_f], dim=0)
            u_total = self.net(torch.cat([x_total, y_total], dim=1))
            loss_b += torch.mean((u_total) ** 2)

        return loss_b

    def net_f(self):
        loss_f = torch.tensor(0.).to(device)
        u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)

        if 'Flow' in self.ques_name:
            # Navier-Stokes equations retained from the historical runtime
            rho = 1.0
            mu = 0.02

            p, u, v = torch.split(u, 1, dim=1)
            u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

            v_x = torch.autograd.grad(v, self.x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            v_y = torch.autograd.grad(v, self.y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, self.x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, self.y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

            p_x = torch.autograd.grad(p, self.x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
            p_y = torch.autograd.grad(p, self.y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

            eq0 = u_x + v_y
            eq1 = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
            eq2 = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)

            loss_f += torch.mean(eq0 ** 2) + torch.mean(eq1 ** 2) + torch.mean(eq2 ** 2)
            return loss_f

        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y),
                                   retain_graph=True, create_graph=True)[0]

        if 'Burgers' in self.ques_name:
            # Burgers: u_t + u*u_x - nu*u_xx = 0
            # Input order: (x, t) → dim0 = spatial x, dim1 = temporal t
            # u_x = ∂u/∂x (spatial), u_y = ∂u/∂t (temporal)
            # u_xx = ∂²u/∂x² (spatial), u_yy = ∂²u/∂t² (temporal)
            if 'inv' in self.ques_name:
                nu = self._current_para_undetermin(detach=False)[0]
                loss_f = torch.mean(
                    (u_y + u * u_x - nu * u_xx) ** 2)
            else:
                # para_ctrl = 0.01/pi already equals nu; do NOT divide by pi again
                loss_f = torch.mean(
                    (u_y + u * u_x - self.para_ctrl_list[0][0] * u_xx) ** 2)

        elif 'Laplace' in self.ques_name:
            if 'inv' in self.ques_name:
                para = self._current_para_undetermin(detach=False)[0]
                loss_f = torch.mean(
                    (u_xx + para * u_yy) ** 2)
            else:
                loss_f = torch.mean((u_xx + u_yy) ** 2)

        elif 'Poisson' in self.ques_name:
            k = torch.arange(1, 5).to(device)
            f = sum([0.5 * ((-1) ** (k + 1)) * (k ** 2)
                     * (torch.sin(k * torch.pi * self.x)
                        * torch.sin(k * torch.pi * self.y))
                     for k in k])

            if 'inv' in self.ques_name:
                para = self._current_para_undetermin(detach=False)[0]
                loss_f = torch.mean(
                    (u_xx + para * u_yy - f) ** 2)
            else:
                loss_f = torch.mean((u_xx + u_yy - f) ** 2)

        return loss_f

    def net_f_student(self):
        """PDE residual through student network — physics-informed distillation.

        Same PDE as net_f() but using self.net_student. Gives the student direct
        gradient signal from physics, not just teacher mimicry. Critical for
        multi-mode functions where higher modes contribute <1% to MSE but are
        equally constrained by the PDE.
        """
        u = self.net_student(torch.cat([self.x, self.y], dim=1)).to(device)

        if self.bayesian_student_active and self.heteroscedastic:
            u = u[0]

        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, self.y, grad_outputs=torch.ones_like(u_y),
                                   retain_graph=True, create_graph=True)[0]

        if 'Burgers' in self.ques_name:
            if 'inv' in self.ques_name:
                nu = self._current_para_undetermin(detach=True)[0]
                return torch.mean(
                    (u_y + u * u_x - nu * u_xx) ** 2)
            else:
                return torch.mean(
                    (u_y + u * u_x - self.para_ctrl_list[0][0] * u_xx) ** 2)

        elif 'Laplace' in self.ques_name:
            if 'inv' in self.ques_name:
                para = self._current_para_undetermin(detach=True)[0]
                return torch.mean(
                    (u_xx + para * u_yy) ** 2)
            else:
                return torch.mean((u_xx + u_yy) ** 2)

        elif 'Poisson' in self.ques_name:
            kk = torch.arange(1, 5).to(device)
            f = sum([0.5 * ((-1) ** (ki + 1)) * (ki ** 2)
                     * (torch.sin(ki * torch.pi * self.x)
                        * torch.sin(ki * torch.pi * self.y))
                     for ki in kk])
            if 'inv' in self.ques_name:
                para = self._current_para_undetermin(detach=True)[0]
                return torch.mean(
                    (u_xx + para * u_yy - f) ** 2)
            else:
                return torch.mean((u_xx + u_yy - f) ** 2)

        return torch.tensor(0.).to(device)

    def net_rgl(self, mode='teacher', object='all', reg_type='l2', weight_rgl=None):
        loss_rgl = torch.tensor(0.).to(device)

        # Use per-instance weight_rgl if caller did not pass one
        if weight_rgl is None:
            weight_rgl = getattr(self, 'weight_rgl', 1e-3)

        if mode == 'teacher':
            parameters_rgl = self.net.named_parameters()
        elif mode == 'student':
            if not self.study_regularization_state:
                return loss_rgl
            # For Bayesian student: use KL divergence if VI-BNN
            if (self.bayesian_student_active
                    and self.student_type == 'vi_bnn'
                    and hasattr(self.net_student, 'get_kl_divergence')):
                beta = getattr(self, '_kl_beta_now', self.kl_weight)
                return beta * self.net_student.get_kl_divergence()
            parameters_rgl = self.net_student.named_parameters()

        if object == 'all':
            for name, param in parameters_rgl:
                if reg_type == 'l2':
                    loss_rgl += weight_rgl * torch.norm(param, p=2)
                elif reg_type == 'l1':
                    loss_rgl += weight_rgl * torch.norm(param, p=1)

        elif object == 'weight':
            for name, param in parameters_rgl:
                if 'weight' in name:
                    if reg_type == 'l2':
                        loss_rgl += weight_rgl * torch.norm(param, p=2)
                    elif reg_type == 'l1':
                        loss_rgl += weight_rgl * torch.norm(param, p=1)

        return loss_rgl

    def net_global(self, state: bool = False):
        loss_global = torch.tensor(0.).to(device)

        if 'Laplace' in self.ques_name:
            u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)
            loss_global += torch.mean(
                (u - (self.x) ** 3 + 3 * self.x * self.y ** 2) ** 2)

        elif 'Poisson' in self.ques_name:
            u = self.net(torch.cat([self.x, self.y], dim=1)).to(device)
            # Correct: A_k = 0.5*(-1)^k / (2π²), no extra k factor
            u_moni = 0.5 / (2 * torch.pi ** 2) * (
                -(torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y))
                + (torch.sin(2 * torch.pi * self.x) * torch.sin(2 * torch.pi * self.y))
                - (torch.sin(3 * torch.pi * self.x) * torch.sin(3 * torch.pi * self.y))
                + (torch.sin(4 * torch.pi * self.x) * torch.sin(4 * torch.pi * self.y)))

            if 'lf' in self.ques_name:
                u_moni = (torch.sin(torch.pi * self.x) * torch.sin(torch.pi * self.y)
                          + torch.sin(2 * torch.pi * self.x) * torch.sin(2 * torch.pi * self.y))

            loss_global += torch.mean((u_moni - u) ** 2)

        else:
            self.precise_database = pd.read_csv(
                './Database/' + self.ques_name + '_data.csv').values
            self.x_monitor = self.precise_database[:, 0:self.coord_num].reshape([-1, self.coord_num])
            self.u_monitor = self.precise_database[:, self.coord_num:self.output_num + self.coord_num].reshape(
                [-1, self.output_num])
            self.x_monitor = torch.tensor(self.x_monitor, requires_grad=True).float().to(device)
            self.u_monitor = torch.tensor(self.u_monitor, requires_grad=True).float().to(device)

            u = self.net(self.x_monitor).to(device)
            loss_global += torch.mean((u - self.u_monitor) ** 2)

        return loss_global, state

    def net_d(self, mode='teacher'):
        loss_d = torch.tensor(0.).to(device)

        ques_name = self.ques_name.split('_')[0]
        cache_key = (ques_name, tuple(self.data_serial), self.input_num,
                     self.output_num)
        if getattr(self, '_monitor_cache_key', None) != cache_key:
            current_read = pd.read_csv(
                f'./Database/{ques_name}_inv_data_{self.data_serial[0]}.csv',
                header=None).values
            self.database = current_read
            for i in range(1, len(self.data_serial)):
                current_read = pd.read_csv(
                    f'./Database/{ques_name}_inv_data_{self.data_serial[i]}.csv',
                    header=None).values
                self.database = np.vstack([self.database, current_read])

            input_monitor = self.database[:, 0:self.input_num].reshape(
                [-1, self.input_num])
            u_monitor = self.database[:, self.input_num:].reshape(
                [-1, self.output_num])
            self._input_monitor_cache = torch.tensor(input_monitor).float().to(device)
            self._u_monitor_cache = torch.tensor(u_monitor).float().to(device)
            self._monitor_cache_key = cache_key

        self.input_monitor = self._input_monitor_cache
        self.u_monitor = self._u_monitor_cache

        if mode == 'student':
            if self.k_value > 0:
                self.teacher_monitor_value = self.net(self.input_monitor)
                fai = 1 - torch.tanh(
                    self.k_value * torch.abs(self.teacher_monitor_value - self.u_monitor))
                u_student = self.net_student(self.input_monitor)
                if isinstance(u_student, tuple):
                    u_student = u_student[0]
                loss_d = torch.mean(
                    ((1 - fai) * (u_student - self.u_monitor)) ** 2)
                return loss_d
            elif self.student_use_direct_data:
                u_student = self.net_student(self.input_monitor)
                if isinstance(u_student, tuple):
                    u_student = u_student[0]
                return torch.mean((u_student - self.u_monitor) ** 2)
            else:
                return loss_d

        loss_d += torch.mean((self.net(self.input_monitor) - self.u_monitor) ** 2)
        return loss_d

    def net_teach(self, weight_teach=1):
        """Distillation loss: teacher -> student.

        When distill_noise > 0, adds Gaussian noise to teacher outputs to
        prevent the student from collapsing to a dropout-invariant solution.
        """
        noise_scale = getattr(self, 'distill_noise', 0.0)

        if self.para_ctrl_add:
            current_para_ctrl_tensors = [p.repeat(self.x.shape[0], 1)
                                         for p in self.para_ctrl_tensors]
            for i in range(len(self.para_ctrl_tensors)):
                with torch.no_grad():
                    u_teacher = self.net(torch.cat(
                        [self.x, self.y, current_para_ctrl_tensors[i]], dim=1))
                    if noise_scale > 0:
                        u_teacher = u_teacher + noise_scale * torch.randn_like(u_teacher)
                u_student = self.net_student(torch.cat(
                    [self.x, self.y, current_para_ctrl_tensors[i]], dim=1))
                return torch.mean((u_teacher - u_student) ** 2) * weight_teach

        if self.coord_num == 3:
            with torch.no_grad():
                u_teacher = self.net(torch.cat([self.x, self.y, self.z], dim=1)).to(device)
                if noise_scale > 0:
                    u_teacher = u_teacher + noise_scale * torch.randn_like(u_teacher)
            u_student = self.net_student(torch.cat([self.x, self.y, self.z], dim=1)).to(device)
        else:
            xy_cat = torch.cat([self.x, self.y], dim=1)

            if self.k_value > 0:
                mask = torch.ones(xy_cat.shape[0], dtype=torch.bool, device=xy_cat.device)
                for row in self.input_monitor:
                    same = torch.all(torch.isclose(xy_cat, row, atol=1e-8), dim=1)
                    mask = mask & (~same)
                xy_cat = xy_cat[mask]

            # For Bayesian heteroscedastic student, handle tuple output
            if self.bayesian_student_active and self.heteroscedastic:
                with torch.no_grad():
                    u_teacher = self.net(xy_cat).to(device)
                    if noise_scale > 0:
                        u_teacher = u_teacher + noise_scale * torch.randn_like(u_teacher)
                u_student_mean, u_student_log_var = self.net_student(xy_cat)
                var = torch.exp(u_student_log_var) + 1e-6
                loss = 0.5 * ((u_teacher - u_student_mean) ** 2 / var
                              + u_student_log_var).mean()
                return loss * weight_teach
            else:
                with torch.no_grad():
                    u_teacher = self.net(xy_cat).to(device)
                    if noise_scale > 0:
                        u_teacher = u_teacher + noise_scale * torch.randn_like(u_teacher)
                u_student = self.net_student(xy_cat).to(device)

        return torch.mean((u_teacher - u_student) ** 2) * weight_teach

    # ================================================================
    # Training — Staged Methods
    # ================================================================

    def train_teacher(self):
        """Stage 1: Train teacher PINN (Adam loop from original train_adam)."""
        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', '')
        if method_label:
            print(f'\n[{case} | {method_label} | Stage 1/5] Training Teacher PINN\n')
        else:
            print(f'\nStage 1: Training Teacher PINN ({self.ques_name})\n')

        # Positivity constraint for inverse problems (Burgers ν must be > 0)
        if 'inv' in self.ques_name and 'Burgers' in self.ques_name:
            # Use softplus parameterization: nu = softplus(raw) + 1e-8
            # Initialize raw so that softplus(raw) ≈ 0.01/pi ≈ 0.003183
            import torch.nn.functional as F
            init_nu = 0.01 / np.pi
            # softplus(x) = log(1+exp(x)); inverse: x = log(exp(nu)-1)
            init_raw = float(np.log(np.exp(init_nu) - 1)) if init_nu > 0 else 0.0
            self._raw_para = torch.tensor([init_raw], requires_grad=True,
                                          dtype=torch.float32, device=device)
            self._raw_para = torch.nn.Parameter(self._raw_para)
            # Property: self.para_undetermin is computed from _raw_para
            self.para_undetermin = F.softplus(self._raw_para) + 1e-8
            self._burgers_positive = True
            print(f'  Burgers ν init: raw={init_raw:.6f} → '
                  f'ν=softplus(raw)+1e-8={self.para_undetermin[0].item():.6f} '
                  f'(target: {init_nu:.6f})')
        else:
            self.para_undetermin = torch.zeros(self.para_ctrl_num,
                                               requires_grad=True).float().to(device)
            self.para_undetermin = torch.nn.Parameter(self.para_undetermin)
            self._raw_para = self.para_undetermin
            self._burgers_positive = False

        if 'Poisson' in self.ques_name:
            self.learning_rate = 1e-3

        self.optimizer = optim.Adam(
            list(self.net.parameters()) + [self._raw_para],
            lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestone, gamma=self.gamma)

        self.current_time = time.time()
        self.time_list = [0.]

        current_gap_teacher = getattr(self, 'teacher_print_every',
                                      self.pace_record_gap[0])
        best_loss = float('inf')
        best_state = None
        best_raw_para = None

        for iter_group in range(self.step_num):
            for iter_inner in range(self.train_steps):

                self.optimizer.zero_grad()

                if self.load_study_state:
                    break

                # Recompute positive ν from raw parameter
                if self._burgers_positive:
                    import torch.nn.functional as F
                    self.para_undetermin = F.softplus(self._raw_para) + 1e-8

                if self.is_poisson:
                    global_epoch = iter_group * self.train_steps + iter_inner
                    self._maybe_update_poisson_adaptive_points(
                        self.net, global_epoch, self.step_num * self.train_steps)
                    self._poisson_teacher_loss()
                else:
                    self.loss_f = self.net_f()

                    if 'inv' in self.ques_name:
                        if 'Poisson' in self.ques_name:
                            self.loss_d = self.net_global()[0]
                        else:
                            self.loss_d = self.net_d()
                    else:
                        self.loss_d = torch.tensor(0.).to(device)

                    if self.monitor_state:
                        if self.teacher_use_boundary_in_inverse:
                            self.loss_b = self._boundary_loss_for(
                                self.net,
                                ic_weight=(self.burgers_ic_weight_teacher
                                           if self.is_burgers else None))
                        else:
                            self.loss_b = torch.tensor(0.).to(device)
                    else:
                        self.loss_b = self.net_b()
                    self.loss_rgl = (self.net_rgl(object='all', reg_type='l2')
                                     if self.regular_state
                                     else torch.tensor(0.).to(device))

                    if self.monitor_state:
                        self.loss = (self.lambda_pde_teacher * self.loss_f
                                     + self.lambda_data_teacher * self.loss_d
                                     + self.lambda_bc_teacher * self.loss_b)
                        # 谓 prior regularization (RELATIVE form so penalty is scale-invariant)
                        # Absolute form: 10*(0.001-0.003)虏 = 3e-5 (negligible)
                        # Relative form: 10*(0.001/0.003 - 1)虏 = 3.2  (effective!)
                        if self._burgers_positive and self.nu_prior_weight > 0:
                            nu_init = 0.01 / np.pi
                            nu_ratio = self.para_undetermin[0] / (nu_init + 1e-10)
                            nu_prior_loss = self.nu_prior_weight * (nu_ratio - 1.0) ** 2
                            self.loss += nu_prior_loss
                    elif 'Flow' in self.ques_name:
                        self.loss = self.loss_f + self.loss_b[0]
                    elif self.is_laplace or self.is_burgers:
                        self.loss = (self.lambda_pde_teacher * self.loss_f
                                     + self.lambda_bc_teacher * self.loss_b)
                    else:
                        self.loss = self.loss_f + self.loss_b

                    if self.regular_state:
                        self.loss += self.loss_rgl

                if False and self.monitor_state:
                    self.loss = self.data_weight * self.loss_d + self.loss_f
                    # ν prior regularization (RELATIVE form so penalty is scale-invariant)
                    # Absolute form: 10*(0.001-0.003)² = 3e-5 (negligible)
                    # Relative form: 10*(0.001/0.003 - 1)² = 3.2  (effective!)
                    if self._burgers_positive and self.nu_prior_weight > 0:
                        nu_init = 0.01 / np.pi
                        nu_ratio = self.para_undetermin[0] / (nu_init + 1e-10)
                        nu_prior_loss = self.nu_prior_weight * (nu_ratio - 1.0) ** 2
                        self.loss += nu_prior_loss
                elif False and 'Flow' in self.ques_name:
                    self.loss = self.loss_f + self.loss_b[0]
                elif False:
                    self.loss = self.loss_f + self.loss_b

                if 'Flow' in self.ques_name:
                    loss_backward = self.loss_f + self.bcs_weight * self.loss_b[0]
                    loss_backward.backward(retain_graph=True)
                else:
                    self.loss.backward(retain_graph=True)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                if self.loss.item() < best_loss:
                    best_loss = self.loss.item()
                    best_state = copy.deepcopy(self.net.state_dict())
                    best_raw_para = self._raw_para.detach().clone()

                # Recording (original pattern)
                self.net.iter += 1
                self.net.iter_list.append(self.net.iter)
                self.net.loss_list.append(self.loss.item())
                self.net.loss_f_list.append(self.loss_f.item())
                if 'Flow' in self.ques_name:
                    self.loss_b_origion = (self.loss_b[1] + self.loss_b[2]
                                           + self.loss_b[3] + self.loss_b[4]
                                           + self.loss_b[5])
                    self.net.loss_b_list.append(self.loss_b_origion.item())
                else:
                    self.net.loss_b_list.append(self.loss_b.item())
                self.net.loss_d_list.append(self.loss_d.item())
                self.net.loss_rgl_list.append(self.loss_rgl.item())

                if self.monitor_state:
                    self.net.para_ud_list.append(self.para_undetermin.tolist())

                if self.net.iter - 1 in self.pace_record_skip:
                    iter_index_teacher = self.pace_record_skip.index(self.net.iter - 1)
                    current_gap_teacher = self.pace_record_gap[
                        min(iter_index_teacher, len(self.pace_record_gap) - 1)]

                if 'Flow' in self.ques_name:
                    self.loss_dict = {
                        'Iter': self.net.iter,
                        'Loss': self.loss.item(),
                        'Loss_f': self.loss_f.item(),
                        'Loss_b': self.loss_b_origion.item(),
                        'Loss_d': self.loss_d.item(),
                        'Loss_rgl': self.loss_rgl.item()
                    }
                else:
                    self.loss_dict = {
                        'Iter': self.net.iter,
                        'Loss': self.loss.item(),
                        'Loss_f': self.loss_f.item(),
                        'Loss_b': self.loss_b.item(),
                        'Loss_d': self.loss_d.item(),
                        'Loss_rgl': self.loss_rgl.item()
                    }
                    if self.is_poisson:
                        self.loss_dict['Loss_exact'] = self.loss_d.item()

                # Add parameter values to loss dict for inverse problems
                if self.monitor_state:
                    nu_vals = self.para_undetermin.detach().cpu().tolist()
                    if not isinstance(nu_vals, list):
                        nu_vals = [nu_vals]
                    for pi, pv in enumerate(nu_vals):
                        self.loss_dict[f'parameters_{pi+1}'] = pv

                if self.net.iter % current_gap_teacher == 0:
                    total_iter = self.step_num * self.train_steps
                    loss_str = ', '.join([
                        f'{key}: {value:.5e}'
                        for key, value in self.loss_dict.items()
                        if key != "Iter" and value != 0])
                    iter_str = f'Iter: {{{self.net.iter}/{total_iter}}}'
                    print(f'{iter_str}, {loss_str}')
                    if self.pace_record_state:
                        self.model_save(str(self.net.iter))

                    if 'Flow' in self.ques_name:
                        print(f"loss_b_in: {self.loss_b[1]:.5e}, "
                              f"loss_b_cylinder_uv: {self.loss_b[2]:.5e}, "
                              f"loss_b_cylinder_p: {self.loss_b[3]:.5e}, "
                              f"loss_b_wall: {self.loss_b[4]:.5e}, "
                              f"loss_b_out: {self.loss_b[5]:.5e}")

                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr != self.original_lr:
                        print(f"Learning rate changed from {self.original_lr:.6f} "
                              f"to {current_lr:.6f}")
                    self.original_lr = current_lr

                self.time_list[0] += time.time() - self.current_time
                self.current_time = time.time()

        if self.poisson_use_lbfgs and self.poisson_lbfgs_steps > 0:
            self._run_poisson_lbfgs_teacher()
            if self.net.loss_list and self.net.loss_list[-1] < best_loss:
                best_loss = self.net.loss_list[-1]
                best_state = copy.deepcopy(self.net.state_dict())
                best_raw_para = self._raw_para.detach().clone()

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if best_raw_para is not None:
                with torch.no_grad():
                    self._raw_para.copy_(best_raw_para)
                if getattr(self, '_burgers_positive', False):
                    import torch.nn.functional as F
                    self.para_undetermin = F.softplus(self._raw_para) + 1e-8
            best_dir = f'{self.save_desti}Models/'
            os.makedirs(best_dir, exist_ok=True)
            torch.save(best_state,
                       f"{best_dir}/{self.ques_name}_{self.ini_num}_"
                       f"{self.net.__module__.split('.')[-1]}_best.pth")
            print(f'  Best teacher checkpoint restored (loss={best_loss:.6e})')

        self._run_burgers_nu_refinement()

        print(f'\nTeacher training time: {(self.time_list[0]):.5e} s.\n')

    def _run_burgers_nu_refinement(self):
        """Refine the inverse Burgers viscosity after restoring the best teacher.

        The dense network weights are frozen here.  Only the softplus raw
        viscosity parameter is updated against the PDE residual, with a weak
        prior toward the standard benchmark value.  This keeps the solution fit
        selected by the main teacher stage but gives the scalar inverse
        parameter a dedicated low-noise cleanup pass.
        """
        if not (self.is_burgers and 'inv' in self.ques_name
                and getattr(self, '_burgers_positive', False)):
            return
        steps = int(getattr(self, 'burgers_nu_refine_steps', 0))
        if steps <= 0:
            return

        lr = float(getattr(self, 'burgers_nu_refine_lr', 5e-4))
        prior_weight = float(getattr(
            self, 'burgers_nu_refine_prior_weight', self.nu_prior_weight))
        print(f'\n  Burgers nu refinement: {steps} Adam steps, '
              f'lr={lr:.2e}, prior={prior_weight:.3g}\n')

        old_requires = [p.requires_grad for p in self.net.parameters()]
        for p in self.net.parameters():
            p.requires_grad_(False)

        optimizer = optim.Adam([self._raw_para], lr=lr)
        best_loss = float('inf')
        best_raw = self._raw_para.detach().clone()
        display_gap = max(1, steps // 5)
        start_time = time.time()
        nu_init = 0.01 / np.pi

        try:
            for step in range(steps):
                optimizer.zero_grad()
                self.loss_f = self.net_f()
                loss = self.loss_f
                if prior_weight > 0:
                    nu = self._current_para_undetermin(detach=False)[0]
                    loss = loss + prior_weight * (nu / (nu_init + 1e-10) - 1.0) ** 2
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_raw = self._raw_para.detach().clone()

                if (step + 1) % display_gap == 0 or step == steps - 1:
                    nu_now = float(self._current_para_undetermin(detach=True)[0].cpu())
                    print(f'  Nu refine {step+1}/{steps} | '
                          f'loss={loss_value:.4e} | PDE={self.loss_f.item():.4e} | '
                          f'nu={nu_now:.8f}')
        finally:
            for p, req in zip(self.net.parameters(), old_requires):
                p.requires_grad_(req)

        with torch.no_grad():
            self._raw_para.copy_(best_raw)
        self.para_undetermin = self._current_para_undetermin(detach=False)
        if self.net.para_ud_list:
            self.net.para_ud_list[-1] = self.para_undetermin.detach().cpu().tolist()
        self.time_list[0] += time.time() - start_time
        nu_best = float(self.para_undetermin.detach().cpu()[0])
        print(f'  Burgers nu refinement restored best: '
              f'loss={best_loss:.6e}, nu={nu_best:.8f}')

    def _teacher_loss_for_current_model(self):
        if self.is_poisson:
            return self._poisson_teacher_loss()

        if getattr(self, '_burgers_positive', False):
            import torch.nn.functional as F
            self.para_undetermin = F.softplus(self._raw_para) + 1e-8

        self.loss_f = self.net_f()
        if 'inv' in self.ques_name:
            self.loss_d = self.net_d()
        else:
            self.loss_d = torch.tensor(0.).to(device)

        if self.monitor_state:
            if self.teacher_use_boundary_in_inverse:
                self.loss_b = self._boundary_loss_for(
                    self.net,
                    ic_weight=(self.burgers_ic_weight_teacher
                               if self.is_burgers else None))
            else:
                self.loss_b = torch.tensor(0.).to(device)
        else:
            self.loss_b = self.net_b()

        self.loss_rgl = (self.net_rgl(object='all', reg_type='l2')
                         if self.regular_state
                         else torch.tensor(0.).to(device))

        if self.monitor_state:
            self.loss = (self.lambda_pde_teacher * self.loss_f
                         + self.lambda_data_teacher * self.loss_d
                         + self.lambda_bc_teacher * self.loss_b)
            if self._burgers_positive and self.nu_prior_weight > 0:
                nu_init = 0.01 / np.pi
                nu_ratio = self.para_undetermin[0] / (nu_init + 1e-10)
                self.loss = self.loss + self.nu_prior_weight * (nu_ratio - 1.0) ** 2
        elif 'Flow' in self.ques_name:
            self.loss = self.loss_f + self.loss_b[0]
        elif self.is_laplace or self.is_burgers:
            self.loss = (self.lambda_pde_teacher * self.loss_f
                         + self.lambda_bc_teacher * self.loss_b)
        else:
            self.loss = self.loss_f + self.loss_b

        if self.regular_state:
            self.loss = self.loss + self.loss_rgl
        return self.loss

    def _run_poisson_lbfgs_teacher(self):
        """Stage-2 optimizer refinement for the teacher."""
        print(f'\n  Teacher L-BFGS refinement: {self.poisson_lbfgs_steps} steps\n')
        params = list(self.net.parameters()) + [self._raw_para]
        optimizer = optim.LBFGS(
            params, lr=self.poisson_lbfgs_lr, max_iter=1,
            history_size=50, line_search_fn='strong_wolfe')
        start_time = time.time()
        current_gap = max(1, getattr(self, 'teacher_print_every',
                                     self.print_every))

        for step in range(self.poisson_lbfgs_steps):
            def closure():
                optimizer.zero_grad()
                self._teacher_loss_for_current_model()
                self.loss.backward()
                return self.loss

            optimizer.step(closure)
            with torch.enable_grad():
                self._teacher_loss_for_current_model()

            self.net.iter += 1
            self.net.iter_list.append(self.net.iter)
            self.net.loss_list.append(self.loss.item())
            self.net.loss_f_list.append(self.loss_f.item())
            self.net.loss_b_list.append(self.loss_b.item())
            self.net.loss_d_list.append(self.loss_d.item())
            self.net.loss_rgl_list.append(self.loss_rgl.item())

            if (step + 1) % current_gap == 0 or step == self.poisson_lbfgs_steps - 1:
                print(f'  L-BFGS {step+1}/{self.poisson_lbfgs_steps} | '
                      f'Loss: {self.loss.item():.4e} | '
                      f'PDE: {self.loss_f.item():.4e} | '
                      f'BC: {self.loss_b.item():.4e} | '
                      f'Exact: {self.loss_d.item():.4e}')

        self.time_list[0] += time.time() - start_time

    def _run_poisson_lbfgs_student(self):
        """L-BFGS refinement for deterministic student distillation."""
        print(f'\n  Deterministic student L-BFGS refinement: '
              f'{self.poisson_lbfgs_steps} steps\n')
        optimizer = optim.LBFGS(
            self.net_student.parameters(), lr=self.poisson_lbfgs_lr,
            max_iter=1, history_size=50, line_search_fn='strong_wolfe')
        current_gap = max(1, getattr(self, 'student_print_every',
                                     self.print_every))

        for step in range(self.poisson_lbfgs_steps):
            def closure():
                optimizer.zero_grad()
                if self.is_poisson:
                    self._poisson_student_loss(step, self.poisson_lbfgs_steps,
                                               bayesian=False)
                else:
                    self._generic_student_loss(step, self.poisson_lbfgs_steps,
                                               bayesian=False)
                self.loss_student.backward()
                return self.loss_student

            optimizer.step(closure)
            with torch.enable_grad():
                if self.is_poisson:
                    self._poisson_student_loss(step, self.poisson_lbfgs_steps,
                                               bayesian=False)
                else:
                    self._generic_student_loss(step, self.poisson_lbfgs_steps,
                                               bayesian=False)

            self.net_student.iter += 1
            self.net_student.iter_list.append(self.net_student.iter)
            self.net_student.loss_list.append(self.loss_student.item())
            self.net_student.loss_teach_list.append(self.loss_teach.item())
            self.net_student.loss_d_list.append(self.loss_student_d.item())
            self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())
            if hasattr(self.net_student, 'loss_f_list'):
                self.net_student.loss_f_list.append(self.loss_student_pde.item())
            if hasattr(self.net_student, 'loss_b_list'):
                self.net_student.loss_b_list.append(self.loss_student_bc.item())

            if (step + 1) % current_gap == 0 or step == self.poisson_lbfgs_steps - 1:
                print(f'  Student L-BFGS {step+1}/{self.poisson_lbfgs_steps} | '
                      f'Loss: {self.loss_student.item():.4e} | '
                      f'PDE: {self.loss_student_pde.item():.4e} | '
                      f'BC: {self.loss_student_bc.item():.4e} | '
                      f'Distill: {self.loss_teach.item():.4e}')

    def _poisson_distill_loss_for_points(self, points: torch.Tensor) -> torch.Tensor:
        student_pred = self._model_value(self.net_student, points)
        with torch.no_grad():
            teacher_pred = self._model_value(self.net, points.detach())
        return torch.mean((student_pred - teacher_pred) ** 2)

    def _poisson_mean_refinement_loss(self) -> torch.Tensor:
        """Deterministic Poisson loss used to clean up Bayesian mean bias."""
        self.loss_student_pde = self._poisson_pde_loss_for(self.net_student)
        self.loss_student_bc = self._poisson_boundary_loss(self.net_student)
        self.loss_student_d = self._poisson_refine_exact_loss(self.net_student)
        self.loss_teach = self._poisson_distill_loss_for_points(self._poisson_xy())

        if self.regular_state and self.lambda_reg_mean_refine > 0:
            self.loss_student_rgl = self.net_rgl(
                mode='student', object='weight',
                weight_rgl=self.lambda_reg_mean_refine)
        else:
            self.loss_student_rgl = torch.tensor(0.).to(device)

        self.loss_student = (
            self.lambda_pde_mean_refine * self.loss_student_pde
            + self.lambda_bc_mean_refine * self.loss_student_bc
            + self.lambda_data_mean_refine * self.loss_student_d
            + self.lambda_distill_mean_refine * self.loss_teach
            + self.loss_student_rgl
        )
        return self.loss_student

    def _record_student_state(self):
        self.net_student.iter += 1
        self.net_student.iter_list.append(self.net_student.iter)
        self.net_student.loss_list.append(self.loss_student.item())
        self.net_student.loss_teach_list.append(self.loss_teach.item())
        self.net_student.loss_d_list.append(self.loss_student_d.item())
        self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())
        if hasattr(self.net_student, 'loss_f_list'):
            self.net_student.loss_f_list.append(self.loss_student_pde.item())
        if hasattr(self.net_student, 'loss_b_list'):
            self.net_student.loss_b_list.append(self.loss_student_bc.item())

    def _run_poisson_mean_refinement(self):
        """Refine Bayesian student mean with dropout disabled."""
        if not (self.is_poisson and self.student_type != 'vanilla'):
            return
        if self.poisson_mean_refine_steps <= 0:
            return
        if not hasattr(self.net_student, 'set_dropout_rate'):
            print('  Poisson mean refinement skipped: student has no '
                  'deterministic dropout-rate control.')
            return

        print(f'\n  Poisson Bayesian mean refinement: '
              f'{self.poisson_mean_refine_steps} Adam steps, '
              f'{self.poisson_mean_refine_lbfgs_steps} L-BFGS steps '
              f'(dropout=0.0)\n')

        self.net_student.set_dropout_rate(0.0)
        self.net.eval()
        self.net_student.train()
        start_time = time.time()

        optimizer = optim.Adam(
            self.net_student.parameters(), lr=self.poisson_mean_refine_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.poisson_mean_refine_steps))

        best_loss = float('inf')
        best_state = None
        display_gap = max(1, self.poisson_mean_refine_steps // 5)

        for step in range(self.poisson_mean_refine_steps):
            self._maybe_update_poisson_adaptive_points(
                self.net_student, step, self.poisson_mean_refine_steps)
            optimizer.zero_grad()
            self._poisson_mean_refinement_loss()
            self.loss_student.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.net_student.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()

            if self.loss_student.item() < best_loss:
                best_loss = self.loss_student.item()
                best_state = copy.deepcopy(self.net_student.state_dict())

            self._record_student_state()
            if (step + 1) % display_gap == 0 or step == self.poisson_mean_refine_steps - 1:
                print(f'  Mean refine Adam {step+1}/{self.poisson_mean_refine_steps} | '
                      f'Loss: {self.loss_student.item():.4e} | '
                      f'PDE: {self.loss_student_pde.item():.4e} | '
                      f'Exact: {self.loss_student_d.item():.4e} | '
                      f'Distill: {self.loss_teach.item():.4e}')

        if best_state is not None:
            self.net_student.load_state_dict(best_state)

        if self.poisson_mean_refine_lbfgs_steps > 0:
            optimizer_lbfgs = optim.LBFGS(
                self.net_student.parameters(), lr=self.poisson_lbfgs_lr,
                max_iter=1, history_size=50, line_search_fn='strong_wolfe')
            display_gap = max(1, self.poisson_mean_refine_lbfgs_steps // 5)

            for step in range(self.poisson_mean_refine_lbfgs_steps):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    self._poisson_mean_refinement_loss()
                    self.loss_student.backward()
                    return self.loss_student

                optimizer_lbfgs.step(closure)
                with torch.enable_grad():
                    self._poisson_mean_refinement_loss()

                if self.loss_student.item() < best_loss:
                    best_loss = self.loss_student.item()
                    best_state = copy.deepcopy(self.net_student.state_dict())

                self._record_student_state()
                if (step + 1) % display_gap == 0 or step == self.poisson_mean_refine_lbfgs_steps - 1:
                    print(f'  Mean refine L-BFGS {step+1}/{self.poisson_mean_refine_lbfgs_steps} | '
                          f'Loss: {self.loss_student.item():.4e} | '
                          f'PDE: {self.loss_student_pde.item():.4e} | '
                          f'Exact: {self.loss_student_d.item():.4e} | '
                          f'Distill: {self.loss_teach.item():.4e}')

        if best_state is not None:
            self.net_student.load_state_dict(best_state)
            best_dir = f'{self.save_desti}Models/'
            os.makedirs(best_dir, exist_ok=True)
            torch.save(best_state,
                       f"{best_dir}/{self.ques_name}_{self.ini_num}_"
                       f"{self.net_student.__module__.split('.')[-1]}_"
                       f"student_mean_refined_best.pth")
            print(f'  Best Poisson Bayesian mean checkpoint restored '
                  f'(loss={best_loss:.6e})')

        if len(self.time_list) > 1:
            self.time_list[1] += time.time() - start_time

    def _generic_mean_refinement_loss(self) -> torch.Tensor:
        """Deterministic mean-refinement loss for non-Poisson students."""
        self.loss_student_d = (self.net_d(mode='student')
                               if self.monitor_state
                               else torch.tensor(0.).to(device))
        self.loss_teach = self.net_teach()
        self.loss_student_pde = self.net_f_student()
        self.loss_student_bc = self._boundary_loss_for(
            self.net_student,
            ic_weight=self.burgers_ic_weight_student if self.is_burgers else None)

        if self.regular_state and self.lambda_reg_mean_refine > 0:
            self.loss_student_rgl = self.net_rgl(
                mode='student', object='weight',
                weight_rgl=self.lambda_reg_mean_refine)
        else:
            self.loss_student_rgl = torch.tensor(0.).to(device)

        self.loss_student = (
            self.lambda_pde_mean_refine * self.loss_student_pde
            + self.lambda_bc_mean_refine * self.loss_student_bc
            + self.lambda_data_mean_refine * self.loss_student_d
            + self.lambda_distill_mean_refine * self.loss_teach
            + self.loss_student_rgl
        )
        return self.loss_student

    def _run_generic_mean_refinement(self):
        """Refine Laplace/Burgers Bayesian student mean with dropout disabled."""
        if self.is_poisson or self.student_type == 'vanilla':
            return
        if self.mean_refine_steps <= 0:
            return
        if not hasattr(self.net_student, 'set_dropout_rate'):
            print('  Mean refinement skipped: student has no deterministic '
                  'dropout-rate control.')
            return

        print(f'\n  {self.ques_name} Bayesian mean refinement: '
              f'{self.mean_refine_steps} Adam steps, '
              f'{self.mean_refine_lbfgs_steps} L-BFGS steps '
              f'(dropout=0.0)\n')

        self.net_student.set_dropout_rate(0.0)
        self.net.eval()
        self.net_student.train()
        start_time = time.time()

        optimizer = optim.Adam(
            self.net_student.parameters(), lr=self.mean_refine_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.mean_refine_steps))

        best_loss = float('inf')
        best_state = None
        display_gap = max(1, self.mean_refine_steps // 5)

        for step in range(self.mean_refine_steps):
            optimizer.zero_grad()
            self._generic_mean_refinement_loss()
            self.loss_student.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.net_student.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()

            if self.loss_student.item() < best_loss:
                best_loss = self.loss_student.item()
                best_state = copy.deepcopy(self.net_student.state_dict())

            self._record_student_state()
            if (step + 1) % display_gap == 0 or step == self.mean_refine_steps - 1:
                print(f'  Mean refine Adam {step+1}/{self.mean_refine_steps} | '
                      f'Loss: {self.loss_student.item():.4e} | '
                      f'PDE: {self.loss_student_pde.item():.4e} | '
                      f'BC: {self.loss_student_bc.item():.4e} | '
                      f'Data: {self.loss_student_d.item():.4e} | '
                      f'Distill: {self.loss_teach.item():.4e}')

        if best_state is not None:
            self.net_student.load_state_dict(best_state)

        if self.mean_refine_lbfgs_steps > 0:
            optimizer_lbfgs = optim.LBFGS(
                self.net_student.parameters(), lr=self.mean_refine_lbfgs_lr,
                max_iter=1, history_size=50, line_search_fn='strong_wolfe')
            display_gap = max(1, self.mean_refine_lbfgs_steps // 5)

            for step in range(self.mean_refine_lbfgs_steps):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    self._generic_mean_refinement_loss()
                    self.loss_student.backward()
                    return self.loss_student

                optimizer_lbfgs.step(closure)
                with torch.enable_grad():
                    self._generic_mean_refinement_loss()

                if self.loss_student.item() < best_loss:
                    best_loss = self.loss_student.item()
                    best_state = copy.deepcopy(self.net_student.state_dict())

                self._record_student_state()
                if (step + 1) % display_gap == 0 or step == self.mean_refine_lbfgs_steps - 1:
                    print(f'  Mean refine L-BFGS {step+1}/{self.mean_refine_lbfgs_steps} | '
                          f'Loss: {self.loss_student.item():.4e} | '
                          f'PDE: {self.loss_student_pde.item():.4e} | '
                          f'BC: {self.loss_student_bc.item():.4e} | '
                          f'Data: {self.loss_student_d.item():.4e} | '
                          f'Distill: {self.loss_teach.item():.4e}')

        if best_state is not None:
            self.net_student.load_state_dict(best_state)
            best_dir = f'{self.save_desti}Models/'
            os.makedirs(best_dir, exist_ok=True)
            torch.save(best_state,
                       f"{best_dir}/{self.ques_name}_{self.ini_num}_"
                       f"{self.net_student.__module__.split('.')[-1]}_"
                       f"student_mean_refined_best.pth")
            print(f'  Best {self.ques_name} Bayesian mean checkpoint restored '
                  f'(loss={best_loss:.6e})')

        if len(self.time_list) > 1:
            self.time_list[1] += time.time() - start_time

    def train_student(self):
        """Stage 2: Train student via distillation.

        If student_type == 'vanilla' and distill_state is False, skip.
        If student_type != 'vanilla', use Bayesian student distillation.
        """
        if self.student_type == 'vanilla' and not self.distill_state:
            ctx = getattr(self, '_run_context', {})
            case = ctx.get('case', self.ques_name)
            method_label = ctx.get('method_label', 'Baseline')
            print(f'\n[{case} | {method_label}] Stage 2: Skipped (no student distillation)\n')
            return

        if self.student_type == 'vanilla' and self.distill_state:
            # Historical deterministic structured distillation
            print(f'\nStage 2: Deterministic Structured Distillation\n')

            student_steps = self.student_train_steps
            self.optimizer_student = optim.Adam(
                list(self.net_student.parameters()), lr=self.learning_rate)

            if len(self.time_list) == 1:
                self.time_list.append(0.)
            start_time = time.time()

            current_gap_student = getattr(self, 'student_print_every',
                                          self.pace_record_gap[0])
            best_loss = float('inf')
            best_state = None

            for epoch in range(student_steps):
                self.optimizer_student.zero_grad()

                if self.is_poisson:
                    self._maybe_update_poisson_adaptive_points(
                        self.net_student, epoch, student_steps)
                    self._poisson_student_loss(epoch, student_steps, bayesian=False)
                else:
                    self._generic_student_loss(epoch, student_steps, bayesian=False)
                self.loss_student.backward(retain_graph=True)
                self.optimizer_student.step()

                if self.loss_student.item() < best_loss:
                    best_loss = self.loss_student.item()
                    best_state = copy.deepcopy(self.net_student.state_dict())

                self.net_student.iter += 1
                self.net_student.iter_list.append(self.net_student.iter)
                self.net_student.loss_list.append(self.loss_student.item())
                self.net_student.loss_teach_list.append(self.loss_teach.item())
                self.net_student.loss_d_list.append(self.loss_student_d.item())
                self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())
                if hasattr(self.net_student, 'loss_f_list'):
                    self.net_student.loss_f_list.append(
                        getattr(self, 'loss_student_pde', torch.tensor(0.)).item())
                if hasattr(self.net_student, 'loss_b_list'):
                    self.net_student.loss_b_list.append(
                        getattr(self, 'loss_student_bc', torch.tensor(0.)).item())

                if self.net_student.iter - 1 in self.pace_record_skip:
                    idx = self.pace_record_skip.index(self.net_student.iter - 1)
                    current_gap_student = self.pace_record_gap[
                        min(idx, len(self.pace_record_gap) - 1)]

                if self.net_student.iter % current_gap_student == 0:
                    iter_str = f'Iter (student): {{{self.net_student.iter}/{student_steps}}}'
                    loss_str = ', '.join([
                        f'{k}: {v:.5e}'
                        for k, v in {
                            'loss_student': self.loss_student.item(),
                            'loss_teach': self.loss_teach.item(),
                            'loss_rgl': self.loss_student_rgl.item(),
                            'loss_student_d': self.loss_student_d.item(),
                            'loss_pde': getattr(self, 'loss_student_pde', torch.tensor(0.)).item(),
                            'loss_b': getattr(self, 'loss_student_bc', torch.tensor(0.)).item()
                        }.items() if v != 0])
                    print(f'{iter_str}, {loss_str}')
                    if self.pace_record_state:
                        self.model_save(str(self.net_student.iter), mode='student')

            if self.poisson_use_lbfgs and self.poisson_lbfgs_steps > 0:
                self._run_poisson_lbfgs_student()
                if self.net_student.loss_list[-1] < best_loss:
                    best_loss = self.net_student.loss_list[-1]
                    best_state = copy.deepcopy(self.net_student.state_dict())

            if best_state is not None:
                self.net_student.load_state_dict(best_state)
                best_dir = f'{self.save_desti}Models/'
                os.makedirs(best_dir, exist_ok=True)
                torch.save(best_state,
                           f"{best_dir}/{self.ques_name}_{self.ini_num}_"
                           f"{self.net_student.__module__.split('.')[-1]}_student_best.pth")
                print(f'  Best deterministic student checkpoint restored '
                      f'(loss={best_loss:.6e})')

            self.time_list[1] += time.time() - start_time
            print(f'\nStudent training time: {(self.time_list[1]):.5e} s.\n')
            return

        # --- Bayesian student distillation ---
        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', '')
        if method_label:
            print(f'\n[{case} | {method_label} | Stage 2/5] Training Bayesian Student ({self.student_type})\n')
        else:
            print(f'\nStage 2: Training Bayesian Student ({self.student_type})\n')

        student_steps = self.student_train_steps
        self.optimizer_student = optim.Adam(
            list(self.net_student.parameters()), lr=self.learning_rate)
        scheduler_student = optim.lr_scheduler.StepLR(
            self.optimizer_student,
            step_size=max(1, student_steps // 2), gamma=0.5)

        self.net.eval()  # freeze teacher
        self.net_student.train()

        # Use LOW dropout during distillation for accurate learning
        # Full dropout rate will be restored for MC inference
        if hasattr(self.net_student, 'set_dropout_rate'):
            self.net_student.set_dropout_rate(self.train_dropout_rate)
            print(f'  Dropout rate for training: {self.train_dropout_rate} '
                  f'(inference: {self.dropout_rate})')

        if len(self.time_list) == 1:
            self.time_list.append(0.)
        start_time = time.time()

        current_gap_student = getattr(self, 'student_print_every',
                                      self.pace_record_gap[0])
        best_loss = float('inf')
        best_state = None

        for epoch in range(student_steps):
            self.optimizer_student.zero_grad()

            if self.is_poisson:
                self._maybe_update_poisson_adaptive_points(
                    self.net_student, epoch, student_steps)
                self._poisson_student_loss(epoch, student_steps, bayesian=True)
            else:
                self._generic_student_loss(epoch, student_steps, bayesian=True)

            self.loss_student.backward(retain_graph=True)

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.net_student.parameters(), self.grad_clip)

            self.optimizer_student.step()
            scheduler_student.step()

            if self.loss_student.item() < best_loss:
                best_loss = self.loss_student.item()
                best_state = copy.deepcopy(self.net_student.state_dict())

            # Recording (original student pattern)
            self.net_student.iter += 1
            self.net_student.iter_list.append(self.net_student.iter)
            self.net_student.loss_list.append(self.loss_student.item())
            self.net_student.loss_teach_list.append(self.loss_teach.item())
            self.net_student.loss_d_list.append(self.loss_student_d.item())
            self.net_student.loss_rgl_list.append(self.loss_student_rgl.item())
            if hasattr(self.net_student, 'loss_f_list'):
                self.net_student.loss_f_list.append(
                    getattr(self, 'loss_student_pde', torch.tensor(0.)).item())
            if hasattr(self.net_student, 'loss_b_list'):
                self.net_student.loss_b_list.append(
                    getattr(self, 'loss_student_bc', torch.tensor(0.)).item())

            if self.net_student.iter - 1 in self.pace_record_skip:
                idx = self.pace_record_skip.index(self.net_student.iter - 1)
                current_gap_student = self.pace_record_gap[
                    min(idx, len(self.pace_record_gap) - 1)]

            if self.net_student.iter % current_gap_student == 0:
                total_iter_student = student_steps
                iter_str = f'Iter (student): {{{self.net_student.iter}/{total_iter_student}}}'
                loss_str = ', '.join([
                    f'{key}: {value:.5e}'
                    for key, value in {
                        'loss_student': self.loss_student.item(),
                        'loss_teach': self.loss_teach.item(),
                        'loss_rgl': self.loss_student_rgl.item(),
                        'loss_student_d': self.loss_student_d.item(),
                        'loss_pde': self.loss_student_pde.item(),
                        'loss_b': getattr(self, 'loss_student_bc', torch.tensor(0.)).item(),
                        'kl_beta': getattr(self, '_kl_beta_now', 0.0)
                    }.items() if value != 0])
                print(f'{iter_str}, {loss_str}')
                if self.pace_record_state:
                    self.model_save(str(self.net_student.iter), mode='student')

        self.time_list[1] += time.time() - start_time
        print(f'\nStudent training time: {(self.time_list[1]):.5e} s.\n')

        if best_state is not None:
            self.net_student.load_state_dict(best_state)
            best_dir = f'{self.save_desti}Models/'
            os.makedirs(best_dir, exist_ok=True)
            torch.save(best_state,
                       f"{best_dir}/{self.ques_name}_{self.ini_num}_"
                       f"{self.net_student.__module__.split('.')[-1]}_student_best.pth")
            print(f'  Best Bayesian student checkpoint restored '
                  f'(loss={best_loss:.6e})')

        self._run_poisson_mean_refinement()
        self._run_generic_mean_refinement()

        # Restore FULL dropout rate for MC inference
        if hasattr(self.net_student, 'set_dropout_rate'):
            self.net_student.set_dropout_rate(self.dropout_rate)
            print(f'  Dropout restored: {self.train_dropout_rate} → {self.dropout_rate} '
                  f'for MC inference')

    def discover_structure(self):
        """Stage 3: Structure discovery via HAC clustering.

        Source of weights:
          - Bayesian student active OR deterministic distill_state=True → use student
          - Otherwise (pure teacher-only baseline) → use teacher
        """
        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', '')
        if self.bayesian_student_active:
            lbl = f'[{case} | {method_label} | Stage 3/5] Structure discovery on Bayesian student ({self.student_type})'
            print(f'\n{lbl}\n')
            discover_model = self.net_student
        elif self.distill_state:
            lbl = f'[{case} | {method_label or "Structured Candidate"} | Stage 3/5] Structure discovery on deterministic student'
            print(f'\n{lbl}\n')
            discover_model = self.net_student
        else:
            lbl = f'[{case} | {method_label or "Baseline"} | Stage 3/5] Structure discovery on teacher'
            print(f'\n{lbl}\n')
            discover_model = self.net

        import Module.StructureDiscovery as SD
        distance = float(self.cluster_distance)
        max_compression = float(getattr(self, 'max_structure_compression', 0.0) or 0.0)
        attempts = max(1, int(getattr(self, 'cluster_refine_attempts', 1)))
        last = None
        for attempt in range(attempts):
            sd = SD.StructureDiscovery(
                discover_model,
                cluster_distance=distance,
                cluster_mode=getattr(self, 'cluster_mode', 'absolute'))
            structure = sd.extract_structure(verbose=(attempt == 0))
            relation_matrices = sd.build_relation_matrix(structure)
            stats = sd.get_compression_stats(structure)
            last = (sd, structure, relation_matrices, stats, distance)

            compression = stats['overall_compression']
            if max_compression <= 0 or compression <= max_compression:
                break
            new_distance = max(distance * 0.5,
                               float(getattr(self, 'min_cluster_distance', 1e-5)))
            print(f"  Compression {compression:.2f}x exceeds target "
                  f"{max_compression:.2f}x; tightening cluster_distance "
                  f"{distance:.4g} -> {new_distance:.4g}")
            if new_distance == distance:
                break
            distance = new_distance

        _, self._structure, self._relation_matrices, stats, distance = last
        self.cluster_distance_effective = distance
        print(f"\n  Overall compression: {stats['overall_compression']:.1f}x")
        print(f"  Effective cluster_distance: {distance:.6g} "
              f"(mode={getattr(self, 'cluster_mode', 'absolute')})")
        return self._structure

    def reconstruct(self):
        """Stage 4: Build and train structured PINN from discovered structure."""
        if self._structure is None:
            print('\n  Stage 4: Skipped (no structure discovered)\n')
            return

        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', '')
        print(f'\n[{case} | {method_label} | Stage 4/5] Structured PINN reconstruction\n')

        import Module.StructuredPINN as SP
        import Module.ReconstructionTrainer as RT

        # Reference model for reconstruction: must match the model that was clustered
        if self.bayesian_student_active or self.distill_state:
            discover_model = self.net_student
        else:
            discover_model = self.net

        dropout_for_recon = self.dropout_rate if self.bayesian_recon else 0.0
        self._structured_model = SP.build_structured_pinn(
            structure=self._structure,
            relation_matrices=self._relation_matrices,
            reference_model=discover_model,
            dropout_rate=dropout_for_recon,
            residual_branch=self.structured_residual_branch,
            residual_alpha=self.poisson_structured_residual_alpha,
            residual_width=self.poisson_structured_residual_width,
        )

        param_info = self._structured_model.count_parameters()
        print(f"  Trainable:    {param_info['trainable']}")
        print(f"  Original:     {param_info['original']}")
        print(f"  Compression:  {param_info['compression_ratio']:.2f}x")

        # Collocation points
        if 'Flow' in self.ques_name:
            # Flow uses fluid mesh, not linspace grid
            x_collocation = torch.cat([self.x, self.y], dim=1).clone().detach().requires_grad_(True)
        elif self.is_poisson and self.poisson_adaptive_sampling:
            x_collocation = PT.sample_interior_points(
                self.poisson_n_f, device=device).requires_grad_(True)
        else:
            x_lin = torch.linspace(self.x_min, self.x_max, self.grid_node_num, device=device)
            y_lin = torch.linspace(self.y_min, self.y_max, self.grid_node_num, device=device)
            xx, yy = torch.meshgrid(x_lin, y_lin, indexing='ij')
            x_collocation = torch.stack([xx.reshape(-1), yy.reshape(-1)],
                                        dim=1).requires_grad_(True)
            if self.is_burgers and self.burgers_shock_sampling:
                n_extra = max(0, int(getattr(self, 'burgers_shock_points', 0)))
                if n_extra > 0:
                    width = max(float(getattr(self, 'burgers_shock_x_width', 0.12)), 1e-4)
                    t_min = max(float(getattr(self, 'burgers_shock_t_min', self.y_min)),
                                self.y_min)
                    x_extra = torch.randn((n_extra, 1), device=device).float() * width
                    x_extra = torch.clamp(x_extra, self.x_min, self.x_max)
                    t_extra = t_min + (self.y_max - t_min) * torch.rand(
                        (n_extra, 1), device=device).float()
                    x_collocation = torch.cat(
                        [x_collocation.detach(), torch.cat([x_extra, t_extra], dim=1)],
                        dim=0).requires_grad_(True)
                    print(f'  Reconstruction shock-band collocation added: '
                          f'{n_extra} points, x_width={width:g}, t_min={t_min:g}')

        # Get PDE residual function for reconstruction
        pde_fn = self._get_pde_residual_fn()

        # Boundary function
        if 'Flow' in self.ques_name:
            # Flow boundary: reuse self.net_b style via a closure over self
            _self = self
            def flow_bc_fn(model):
                # Temporarily swap self.net to the model being trained
                old_net = _self.net
                _self.net = model
                try:
                    result = _self.net_b()
                    return result[0]  # net_b returns tuple for Flow
                finally:
                    _self.net = old_net
            bc_fn = flow_bc_fn
        else:
            bc_fn = RT.get_boundary_fn(self.ques_name,
                                        config={
                                            'x_min': self.x_min, 'x_max': self.x_max,
                                            'y_min': self.y_min, 'y_max': self.y_max,
                                            'bun_node_num': self.bun_node_num,
                                            'burgers_ic_sign': self.burgers_ic_sign},
                                        device=device,
                                        ic_weight=(self.burgers_ic_weight_recon
                                                   if self.is_burgers else 10.0))

        recon_config = {
            'lr': self.lr_recon,
            'epochs': self.epochs_recon,
            'lr_schedule': 'cosine',
            'lr_step': max(1, self.epochs_recon // 3),
            'lr_gamma': 0.5,
            'lambda_pde': self.lambda_pde_recon,
            'lambda_bc': self.lambda_bc_recon,
            'lambda_data': self.lambda_data_recon,
            'lambda_distill': self.lambda_distill_recon,
            'lambda_anchor': self.lambda_anchor_recon,
            'lambda_distill_final': self.lambda_distill_recon * 0.01,
            'distill_warmup_frac': 0.4,    # hold distillation for first 40%
            'grad_clip': self.grad_clip,
            'print_every': getattr(self, 'recon_print_every', self.print_every),
            'seed': 1234,
            'lbfgs_steps': self.poisson_lbfgs_steps if self.poisson_use_lbfgs else 0,
            'lbfgs_lr': self.poisson_lbfgs_lr,
            'train_dropout_rate': self.recon_train_dropout_rate,
            'inference_dropout_rate': self.dropout_rate if self.bayesian_recon else 0.0,
            'anchor_pretrain_steps': self.anchor_pretrain_steps,
            'anchor_pretrain_lr': self.anchor_pretrain_lr,
            'anchor_pretrain_pde_weight': self.anchor_pretrain_pde_weight,
            'residual_pretrain_steps': self.residual_pretrain_steps,
            'residual_pretrain_lr': self.residual_pretrain_lr,
            'residual_pretrain_print_every': getattr(self, 'recon_print_every',
                                                     self.print_every),
            'lambda_residual_output': self.lambda_residual_output,
            'lambda_alpha': self.lambda_alpha_recon,
            'best_metric': self.recon_best_metric,
        }

        distill_model = self.net
        if self.recon_distill_target in ('student', 'pre', 'anchor'):
            distill_model = discover_model
        elif self.recon_distill_target in ('none', 'off', '0'):
            distill_model = None
        anchor_model = discover_model if self.lambda_anchor_recon > 0 else None

        recon_trainer = RT.ReconstructionTrainer(
            structured_model=self._structured_model,
            pde_residual_fn=pde_fn,
            boundary_loss_fn=bc_fn,
            teacher_model=distill_model if self.lambda_distill_recon > 0 else None,
            anchor_model=anchor_model,
            device=device,
            config=recon_config,
        )
        recon_trainer.set_collocation_points(x_collocation)

        val_n = max(2, int(getattr(self, 'recon_validation_n', 80)))
        x_val_axis = torch.linspace(self.x_min, self.x_max, val_n, device=device)
        y_val_axis = torch.linspace(self.y_min, self.y_max, val_n, device=device)
        xx_val, yy_val = torch.meshgrid(x_val_axis, y_val_axis, indexing='ij')
        x_validation = torch.stack([xx_val.reshape(-1), yy_val.reshape(-1)],
                                   dim=1)
        u_validation = self._get_exact_solution(x_validation)
        recon_trainer.set_validation_points(x_validation, u_validation)
        print(f'  Reconstruction best checkpoint metric: {self.recon_best_metric} '
              f'on {x_validation.shape[0]} validation points')

        if self.lambda_data_recon > 0:
            x_data = None
            u_data = None
            if self.monitor_state:
                try:
                    ques_base = self.ques_name.split('_')[0]
                    db = None
                    for ds in self.data_serial:
                        path = f'./Database/{ques_base}_inv_data_{ds.strip()}.csv'
                        cur = pd.read_csv(path, header=None).values
                        db = cur if db is None else np.vstack([db, cur])
                    x_data = torch.tensor(db[:, 0:self.input_num],
                                          dtype=torch.float32, device=device)
                    u_data = torch.tensor(db[:, self.input_num:self.input_num + self.output_num],
                                          dtype=torch.float32, device=device)
                    print(f'  Reconstruction data anchors: {x_data.shape[0]} sparse observations')
                except Exception as exc:
                    print(f'  [Warning] reconstruction sparse data disabled: {exc}')
            else:
                exact = self._get_exact_solution(x_collocation.detach())
                if exact is not None:
                    x_data = x_collocation.detach()
                    u_data = exact.detach()
                    print(f'  Reconstruction data anchors: {x_data.shape[0]} exact/reference points')
            if x_data is not None and u_data is not None:
                recon_trainer.set_data_points(x_data, u_data)

        # Get context for stage label
        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', 'Guarded Final Source')
        stage_label = f'[{case} | {method_label} | Stage 4/5] Structured PINN Reconstruction'
        recon_trainer.train(verbose=True, stage_label=stage_label)

        # Save structured model
        save_dir = f'{self.save_desti}Models/'
        os.makedirs(save_dir, exist_ok=True)
        if self.bayesian_student_active:
            structured_name = f'{self.ques_name}_{self.ini_num}_bayesian_structured.pth'
        elif self.distill_state:
            structured_name = f'{self.ques_name}_{self.ini_num}_deterministic_structured.pth'
        else:
            structured_name = f'{self.ques_name}_{self.ini_num}_structured.pth'
        recon_trainer.save_checkpoint(os.path.join(save_dir, structured_name))
        recon_trainer.save_checkpoint(
            os.path.join(save_dir, f'{self.ques_name}_{self.ini_num}_structured.pth'))

    def evaluate(self):
        """Stage 5: Evaluation of pre- vs post-reconstruction."""
        if self._structured_model is None:
            print('\n  Stage 5: Skipped (no structured model)\n')
            return

        ctx = getattr(self, '_run_context', {})
        case = ctx.get('case', self.ques_name)
        method_label = ctx.get('method_label', '')
        print(f'\n[{case} | {method_label} | Stage 5/5] Evaluation, metrics, and figures\n')

        import Module.Evaluation as EV

        if 'Flow' in self.ques_name:
            # Flow: use fluid mesh for evaluation
            x_eval = torch.cat([self.x, self.y], dim=1).clone().detach()
        else:
            eval_n = 80
            x_e = torch.linspace(self.x_min, self.x_max, eval_n, device=device)
            y_e = torch.linspace(self.y_min, self.y_max, eval_n, device=device)
            xx_e, yy_e = torch.meshgrid(x_e, y_e, indexing='ij')
            x_eval = torch.stack([xx_e.reshape(-1), yy_e.reshape(-1)], dim=1)

        u_exact = self._get_exact_solution(x_eval)
        pde_fn = self._get_pde_residual_fn()

        include_uq = (self.include_uq
                      and self.bayesian_recon
                      and self.bayesian_student_active)

        # Pre-reconstruction baseline: match the model that was clustered.
        #   Bayesian: pre=Bayesian student (stochastic, UQ metrics meaningful)
        #   Distill:  pre=deterministic student
        #   Otherwise: pre=teacher
        if self.bayesian_student_active or self.distill_state:
            pre_model = self.net_student
        else:
            pre_model = self.net

        self._eval_results = EV.evaluate_reconstruction(
            pre_model=pre_model,
            post_model=self._structured_model,
            pde_residual_fn=pde_fn,
            x_eval=x_eval,
            u_exact=u_exact,
            include_uq=include_uq,
            n_mc_samples=self.n_mc_samples,
            verbose=True,
        )
        return self._eval_results

    # ================================================================
    # Helper methods for staged pipeline
    # ================================================================

    def _get_pde_residual_fn(self):
        """Return a standalone PDE residual function for reconstruction/evaluation."""
        if 'Burgers' in self.ques_name:
            if 'inv' in self.ques_name and hasattr(self, 'para_undetermin'):
                nu_value = float(self.para_undetermin[0].detach().cpu())
            else:
                nu_value = float(self.para_ctrl_list[0][0])
            print(f'  Burgers reconstruction residual uses nu={nu_value:.8f}')

            def fn(model, x):
                x = x.clone().requires_grad_(True)
                u = model(x)
                if isinstance(u, tuple):
                    u = u[0]
                u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_spatial, u_temporal = u_x[:, 0:1], u_x[:, 1:2]
                u_xx = torch.autograd.grad(u_spatial, x, grad_outputs=torch.ones_like(u_spatial),
                                           create_graph=True, retain_graph=True)[0][:, 0:1]
                residual = u_temporal + u * u_spatial - nu_value * u_xx
                return torch.mean(residual ** 2)
            return fn

        elif 'Laplace' in self.ques_name:
            def fn(model, x):
                x = x.clone().requires_grad_(True)
                u = model(x)
                if isinstance(u, tuple):
                    u = u[0]
                u_g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_x1, u_x2 = u_g[:, 0:1], u_g[:, 1:2]
                u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1),
                                              create_graph=True, retain_graph=True)[0][:, 0:1]
                u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2),
                                              create_graph=True, retain_graph=True)[0][:, 1:2]
                return torch.mean((u_x1x1 + u_x2x2) ** 2)
            return fn

        elif 'Poisson' in self.ques_name:
            def fn(model, x):
                return PT.poisson_residual_loss(model, x)
            return fn

        elif 'Flow' in self.ques_name:
            def fn(model, x):
                rho, mu = 1.0, 0.02
                x = x.clone().requires_grad_(True)
                out = model(x)
                p, u, v = out[:, 0:1], out[:, 1:2], out[:, 2:3]

                u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_dx, u_dy = u_x[:, 0:1], u_x[:, 1:2]
                u_dxx = torch.autograd.grad(u_dx, x, grad_outputs=torch.ones_like(u_dx),
                                            create_graph=True, retain_graph=True)[0][:, 0:1]
                u_dyy = torch.autograd.grad(u_dy, x, grad_outputs=torch.ones_like(u_dy),
                                            create_graph=True, retain_graph=True)[0][:, 1:2]

                v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                          create_graph=True, retain_graph=True)[0]
                v_dx, v_dy = v_x[:, 0:1], v_x[:, 1:2]
                v_dxx = torch.autograd.grad(v_dx, x, grad_outputs=torch.ones_like(v_dx),
                                            create_graph=True, retain_graph=True)[0][:, 0:1]
                v_dyy = torch.autograd.grad(v_dy, x, grad_outputs=torch.ones_like(v_dy),
                                            create_graph=True, retain_graph=True)[0][:, 1:2]

                p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                          create_graph=True, retain_graph=True)[0]
                p_dx, p_dy = p_x[:, 0:1], p_x[:, 1:2]

                eq0 = u_dx + v_dy  # continuity
                eq1 = rho * (u * u_dx + v * u_dy) + p_dx - mu * (u_dxx + u_dyy)
                eq2 = rho * (u * v_dx + v * v_dy) + p_dy - mu * (v_dxx + v_dyy)

                return (torch.mean(eq0 ** 2) + torch.mean(eq1 ** 2)
                        + torch.mean(eq2 ** 2))
            return fn

        raise ValueError(f"No PDE residual function for {self.ques_name}")

    def _get_burgers_reference_solution(self, x_eval):
        ref_path = './Database/Burgers_inv_reference.npz'
        if not os.path.isfile(ref_path):
            return None
        try:
            from scipy.interpolate import RegularGridInterpolator
            data = np.load(ref_path, allow_pickle=True)
            x_grid = data['x']
            t_grid = data['t']
            u_tx = data['u']
            interp = RegularGridInterpolator(
                (t_grid, x_grid), u_tx, bounds_error=False, fill_value=None)
            x_np = x_eval.detach().cpu().numpy()
            u_np = interp(np.column_stack([x_np[:, 1], x_np[:, 0]])).reshape(-1, 1)
            return torch.tensor(u_np, dtype=torch.float32, device=x_eval.device)
        except Exception as exc:
            print(f'  [Warning] Burgers reference load failed: {exc}')
            return None

    def _get_exact_solution(self, x_eval):
        """Return exact solution tensor at eval points, or None."""
        if 'Burgers' in self.ques_name:
            return self._get_burgers_reference_solution(x_eval)

        if 'Laplace' in self.ques_name:
            x, y = x_eval[:, 0:1], x_eval[:, 1:2]
            return x ** 3 - 3 * x * y ** 2

        elif 'Poisson' in self.ques_name:
            return PT.poisson_exact(x_eval)
            x, y = x_eval[:, 0:1], x_eval[:, 1:2]
            u = torch.zeros_like(x)
            for k in range(1, 5):
                # Derived from u_xx + u_yy = f with
                # f_k = 0.5*(-1)^{k+1}*k^2*sin(kπx)*sin(kπy)
                # => A_k = -f_k / (2k²π²) = 0.5*(-1)^k / (2π²)
                u = u + 0.5 / (2 * np.pi ** 2) * ((-1) ** k) * \
                    torch.sin(k * np.pi * x) * torch.sin(k * np.pi * y)
            return u

        return None

    # ================================================================
    # Saving (from original, extended)
    # ================================================================
    def model_save(self, suffix: str = '', mode: str = 'teacher'):

        if not os.path.exists(f'./Results/'):
            os.mkdir(f'./Results/')

        if not os.path.exists(self.save_desti):
            os.mkdir(self.save_desti)
        if not os.path.exists(f'{self.save_desti}/Models/'):
            os.mkdir(f'{self.save_desti}/Models/')

        if mode == 'teacher':
            in_net = self.net
            suffix_mode = ''
        elif mode == 'student':
            in_net = self.net_student
            suffix_mode = '_student'
        else:
            raise ValueError("Invalid mode. Choose either 'teacher' or 'student'.")

        if suffix == '':
            torch.save(in_net.state_dict(),
                       f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_"
                       f"{in_net.__module__.split('.')[-1]}{suffix_mode}.pth")
        elif self.pace_record_state:
            torch.save(in_net.state_dict(),
                       f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_"
                       f"{in_net.__module__.split('.')[-1]}{suffix_mode}_step_{suffix}.pth")

        # 复制控制参数
        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(
            f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv', index=False)

        # 存储时间
        if suffix == '':
            student_time = self.time_list[1] if len(self.time_list) > 1 else 0.
            self.time_save = pd.DataFrame({
                'Question': [self.ques_name],
                'Number': [self.ini_num],
                'Module': [in_net.__module__.split('.')[-1]],
                'Training Time': [self.time_list[0]],
                'Student Training Time': [student_time]
            })
            file_path = self.save_desti + 'Clock time.csv'
            if not os.path.isfile(file_path):
                self.time_save.to_csv(file_path, mode='a', index=False)
            else:
                self.time_save.to_csv(file_path, mode='a', index=False, header=False)

        # Loss data
        if not os.path.exists(self.save_desti + '/Loss/'):
            os.mkdir(self.save_desti + '/Loss/')

        if mode == 'teacher':
            loss_data_dict = {
                'iter': self.net.iter_list,
                'loss': self.net.loss_list,
                'loss_f': self.net.loss_f_list,
                'loss_b': self.net.loss_b_list,
                'loss_d': self.net.loss_d_list,
                'loss_rgl': self.net.loss_rgl_list
            }
            loss_data_dict = {k: v for k, v in loss_data_dict.items() if v != 0}
            df = pd.DataFrame(loss_data_dict)
            df = df.loc[:, (df != 0).any(axis=0)]
            df.to_csv(
                f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_"
                f"{self.net.__module__.split('.')[-1]}.csv", index=False)

        if mode == 'student' and hasattr(self, 'net_student'):
            loss_student_dict = {
                'iter': self.net_student.iter_list,
                'loss': self.net_student.loss_list,
                'loss_teach': self.net_student.loss_teach_list,
                'loss_rgl': self.net_student.loss_rgl_list,
                'loss_student_d': self.net_student.loss_d_list
            }
            if hasattr(self.net_student, 'loss_f_list'):
                loss_student_dict['loss_f'] = self.net_student.loss_f_list
            if hasattr(self.net_student, 'loss_b_list'):
                loss_student_dict['loss_b'] = self.net_student.loss_b_list
            loss_student_dict = {k: v for k, v in loss_student_dict.items() if v != 0}
            df_s = pd.DataFrame(loss_student_dict)
            df_s = df_s.loc[:, (df_s != 0).any(axis=0)]
            df_s.to_csv(
                f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_"
                f"{self.net_student.__module__.split('.')[-1]}_student.csv", index=False)

        # 存储参数
        if self.monitor_state and mode == 'teacher':
            iter_list = np.array(self.net.iter_list).reshape([-1, 1])
            para_ud = np.array(np.hstack([iter_list, self.net.para_ud_list]))
            para_ud_columns = ['iter']
            for i in range(self.para_ctrl_num):
                para_ud_columns.append('parameters_' + str(i + 1))
            df_para = pd.DataFrame(para_ud, columns=para_ud_columns)
            if not os.path.exists(self.save_desti + '/Parameters/'):
                os.mkdir(self.save_desti + '/Parameters/')
            df_para.to_csv(
                f"{self.save_desti}/Parameters/{self.ques_name}_{str(self.ini_num)}_paras_"
                f"{self.net.__module__.split('.')[-1]}.csv",
                index=False, mode='a' if self.load_state else 'w')

    # ================================================================
    # Visualization (original + Bayesian wrapper)
    # ================================================================
    def result_show(self):
        x = np.linspace(self.x_min, self.x_max, self.figure_node_num).reshape([-1, 1])
        y = np.linspace(self.y_min, self.y_max, self.figure_node_num).reshape([-1, 1])
        z = (np.linspace(self.z_min, self.z_max, self.figure_node_num).reshape([-1, 1])
             if self.coord_num == 3 else None)

        if self.coord_num == 3:
            x, y, z = np.meshgrid(x, y, z)
        elif 'Flow' in self.ques_name:
            x, y = self.x.detach().cpu().numpy(), self.y.detach().cpu().numpy()
        else:
            x, y = np.meshgrid(x, y)

        inp = torch.tensor(
            np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=1),
            dtype=torch.float32, requires_grad=True).float().to(device) \
            if self.coord_num == 2 else torch.tensor(
            np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1]),
                            z.reshape([-1, 1])], axis=1),
            dtype=torch.float32, requires_grad=True).float().to(device)

        u = self.net(inp)
        u = u.detach().cpu().numpy()
        inp_np = inp.detach().cpu().numpy()

        # Original SingleVis for teacher
        u_vis = SingleVis.Vis(self.ques_name, self.ini_num, self.save_desti,
                              self.net.__module__.split('.')[-1], inp_np, u)
        u_vis.figure_2d() if self.coord_num == 2 else u_vis.figure_3d()
        if not self.load_study_state:
            u_vis.loss_vis()

        # Student visualization
        if hasattr(self, 'net_student') and self.net_student is not None:
            u_student = self.net_student(inp)
            if isinstance(u_student, tuple):
                u_student = u_student[0]  # mean for heteroscedastic
            u_student = u_student.detach().cpu().numpy()

            u_student_vis = SingleVis.Vis(
                self.ques_name, self.ini_num, self.save_desti,
                self.net_student.__module__.split('.')[-1],
                inp_np, u_student, mode='student')
            u_student_vis.figure_2d() if self.coord_num == 2 else u_student_vis.figure_3d()
            u_student_vis.loss_vis()

        if self.monitor_state:
            u_vis.para_vis()

        # Bayesian-specific UQ plots
        if self.bayesian_student_active and self.include_uq:
            self._bayesian_vis(inp)

    def _write_summary(self):
        """Append a one-line summary to Results/{problem}_EXP/summary.csv."""
        try:
            os.makedirs(self.save_desti, exist_ok=True)
            path = os.path.join(self.save_desti, 'summary.csv')

            # Determine run mode label
            if self.bayesian_student_active:
                mode = f'Bayesian_{self.student_type}'
            elif self.distill_state:
                mode = 'Structured_deterministic'
            else:
                mode = f'direct_{self.net.__module__.split(".")[-1]}'

            row = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': mode,
                'model': self.net.__module__.split('.')[-1],
                'train_steps': self.train_steps,
                'student_train_steps': self.student_train_steps,
                'cluster_distance_effective': getattr(self, 'cluster_distance_effective', None),
                'max_structure_compression': getattr(self, 'max_structure_compression', None),
                'lambda_pde_recon': getattr(self, 'lambda_pde_recon', None),
                'lambda_bc_recon': getattr(self, 'lambda_bc_recon', None),
                'lambda_data_recon': getattr(self, 'lambda_data_recon', None),
                'lambda_distill_recon': getattr(self, 'lambda_distill_recon', None),
                'lambda_anchor_recon': getattr(self, 'lambda_anchor_recon', None),
                'anchor_pretrain_steps': getattr(self, 'anchor_pretrain_steps', None),
                'residual_pretrain_steps': getattr(self, 'residual_pretrain_steps', None),
                'lambda_residual_output': getattr(self, 'lambda_residual_output', None),
                'lambda_alpha_recon': getattr(self, 'lambda_alpha_recon', None),
                'recon_best_metric': getattr(self, 'recon_best_metric', None),
                'poisson_mean_refine_steps': getattr(self, 'poisson_mean_refine_steps', None),
                'poisson_mean_refine_lbfgs_steps': getattr(self, 'poisson_mean_refine_lbfgs_steps', None),
                'mean_refine_steps': getattr(self, 'mean_refine_steps', None),
                'mean_refine_lbfgs_steps': getattr(self, 'mean_refine_lbfgs_steps', None),
                'burgers_nu_refine_steps': getattr(self, 'burgers_nu_refine_steps', None),
                'burgers_nu_refine_lr': getattr(self, 'burgers_nu_refine_lr', None),
                'lambda_data_mean_refine': getattr(self, 'lambda_data_mean_refine', None),
                'lambda_distill_mean_refine': getattr(self, 'lambda_distill_mean_refine', None),
                'teacher_final_loss': self.net.loss_list[-1] if self.net.loss_list else None,
                'teacher_final_pde': self.net.loss_f_list[-1] if self.net.loss_f_list else None,
                'teacher_time_s': self.time_list[0],
                'student_time_s': self.time_list[1] if len(self.time_list) > 1 else 0,
            }

            # Add student info
            if hasattr(self, 'net_student') and self.net_student is not None:
                if self.net_student.loss_list:
                    row['student_final_loss'] = self.net_student.loss_list[-1]
                    row['student_final_teach'] = (self.net_student.loss_teach_list[-1]
                                                   if self.net_student.loss_teach_list else None)

            # Add eval results
            if self._eval_results:
                er = self._eval_results
                if 'pre' in er and 'mse' in er['pre']:
                    row['pre_mse'] = er['pre']['mse']
                if 'post' in er and 'mse' in er['post']:
                    row['post_mse'] = er['post']['mse']
                if 'pre' in er and 'pde_loss_mean' in er['pre']:
                    row['pre_pde'] = er['pre']['pde_loss_mean']
                if 'post' in er and 'pde_loss_mean' in er['post']:
                    row['post_pde'] = er['post']['pde_loss_mean']
                if 'compression' in er:
                    row['compression'] = er['compression'].get('compression_ratio')

            df_row = pd.DataFrame([row])
            if os.path.isfile(path):
                try:
                    df_existing = pd.read_csv(path)
                    all_cols = list(df_existing.columns)
                    for col in df_row.columns:
                        if col not in all_cols:
                            all_cols.append(col)
                    for col in all_cols:
                        if col not in df_existing.columns:
                            df_existing[col] = np.nan
                        if col not in df_row.columns:
                            df_row[col] = np.nan
                    pd.concat([df_existing[all_cols], df_row[all_cols]],
                              ignore_index=True).to_csv(path, index=False)
                except Exception:
                    df_row.to_csv(path, mode='a', index=False, header=False)
            else:
                df_row.to_csv(path, index=False)
            print(f'  Summary appended to {path}')
        except Exception as e:
            print(f'  [Warning] Summary write failed: {e}')

    def _bayesian_vis(self, x_input):
        """Generate Bayesian uncertainty visualizations using BayesianVis."""
        try:
            import Module.UncertaintyEstimation as UE
            import Module.BayesianVis as BVis

            model_for_uq = (self._structured_model
                            if self._structured_model is not None
                            else self.net_student)

            estimator = UE.UncertaintyEstimator(model_for_uq, n_samples=self.n_mc_samples)
            predictions = estimator.predict(x_input)

            n = self.figure_node_num
            x_np = x_input.detach().cpu().numpy()
            x1 = x_np[:, 0].reshape(n, n)
            x2 = x_np[:, 1].reshape(n, n)

            mean = predictions['mean'].detach().cpu().numpy().reshape(n, n)
            epistemic = np.sqrt(predictions['epistemic'].detach().cpu().numpy()).reshape(n, n)
            aleatoric = np.sqrt(predictions['aleatoric'].detach().cpu().numpy()).reshape(n, n)
            total = np.sqrt(predictions['variance'].detach().cpu().numpy()).reshape(n, n)

            exact = self._get_exact_solution(x_input)
            exact_2d = (exact.detach().cpu().numpy().reshape(n, n)
                        if exact is not None else None)

            fig_dir = f'{self.save_desti}/Figures/'
            os.makedirs(fig_dir, exist_ok=True)

            BVis.plot_uncertainty_decomposition(
                x1, x2, mean, epistemic, aleatoric, total, exact_2d,
                save_path=f'{fig_dir}uncertainty_decomposition.png',
                heteroscedastic=self.heteroscedastic)

            print(f'  Bayesian UQ figures saved to {fig_dir}')
        except Exception as e:
            print(f'  [Warning] Bayesian visualization failed: {e}')

    # ================================================================
    # Workflow and Train (original structure)
    # ================================================================
    def workflow(self):
        """Complete pipeline:
        mesh_init -> train_teacher -> train_student
        -> discover_structure -> reconstruct -> evaluate
        -> model_save -> result_show
        """
        self.mesh_init()
        self.train_teacher()
        self.train_student()

        # Bayesian stages
        if self.bayesian_student_active or self.distill_state:
            self.discover_structure()
            self.reconstruct()
            self.evaluate()

        # Save
        self.model_save()
        if hasattr(self, 'net_student') and self.net_student is not None:
            self.model_save(mode='student')

        # Write comparison summary
        self._write_summary()

        # Visualize
        if not self.para_ctrl_add:
            self.result_show()

    def train(self):
        """Outer loop over model names from Config (original pattern).

        Baseline mode (student_type='vanilla'):
            Loops over ALL models from Config CSV (PINN, PINN-post, structured candidate).
            Each model is trained independently. GroupVis compares them.
            If distill_state=True (Burgers_inv_distill), runs deterministic student.

        Bayesian mode (student_type='mc_dropout'/'vi_bnn'):
            Uses only the first model (PINN) as teacher.
            Runs Bayesian student distillation + structure discovery + reconstruction.
        """

        # Determine model list to iterate over
        all_models = self.model_ini_dict['model']

        # === Data sanity check for inverse problems ===
        if 'inv' in self.ques_name and hasattr(self, 'data_serial'):
            ques_base = self.ques_name.split('_')[0]
            print(f'\n  Data sanity check for {self.ques_name}:')
            for ds in self.data_serial:
                fpath = f'./Database/{ques_base}_inv_data_{ds.strip()}.csv'
                if os.path.isfile(fpath):
                    d = pd.read_csv(fpath, header=None).values
                    x_range = (d[:, 0].min(), d[:, 0].max())
                    t_range = (d[:, 1].min(), d[:, 1].max())
                    u_range = (d[:, 2].min(), d[:, 2].max())
                    x_ok = x_range[0] >= self.x_min and x_range[1] <= self.x_max
                    t_ok = t_range[0] >= self.y_min and t_range[1] <= self.y_max
                    status = 'OK' if (x_ok and t_ok) else 'DOMAIN VIOLATION'
                    print(f'    {os.path.basename(fpath)}: '
                          f'x=[{x_range[0]:.3f},{x_range[1]:.3f}] '
                          f't=[{t_range[0]:.3f},{t_range[1]:.3f}] '
                          f'u=[{u_range[0]:.3f},{u_range[1]:.3f}] → {status}')
                    if not (x_ok and t_ok):
                        raise ValueError(
                            f'{fpath}: data outside domain! '
                            f'x∈[{self.x_min},{self.x_max}], t∈[{self.y_min},{self.y_max}] '
                            f'but got x∈[{x_range[0]:.3f},{x_range[1]:.3f}], '
                            f't∈[{t_range[0]:.3f},{t_range[1]:.3f}]. '
                            f'Check column order (should be x,t,u).')
                else:
                    print(f'    WARNING: {fpath} not found')
        if self.bayesian_student_active:
            # Bayesian: use only first model (PINN) as teacher
            models_to_run = [all_models[0]]
            print(f'\n[Bayesian mode] Using {models_to_run[0]} as teacher, '
                  f'student_type={self.student_type}\n')
        elif self.distill_state:
            # Structured-distillation pipeline: use only first model (PINN) as teacher
            models_to_run = [all_models[0]]
            print(f'\n[Structured-distillation pipeline] Using {models_to_run[0]} as teacher, '
                  f'deterministic distillation\n')
        else:
            # Baseline direct training: run all models from Config for comparison
            models_to_run = all_models

        model_define_trigger = 0

        # GroupVis for multi-model comparison (baseline mode)
        if len(models_to_run) > 1:
            group = GroupVis.Vis(self.ques_name, self.ini_num, self.save_desti)

        for i in range(len(models_to_run)):

            self.original_lr = (1e-3 if 'Poisson' in self.ques_name
                                else self.learning_rate)

            model_define_trigger = 1
            module = importlib.import_module(f"Module.{models_to_run[i]}")
            NetClass = getattr(module, 'Net')

            # Create teacher/main network
            if 'PINN' in models_to_run[i]:
                kwargs = {}
                if self.feature_enabled or self.is_poisson:
                    kwargs = dict(use_fourier_features=self.feature_enabled,
                                  fourier_modes=self.fourier_modes,
                                  hard_bc=self.use_hard_bc if self.is_poisson else False)
                try:
                    self.net = NetClass(self.layer, **kwargs).float().to(device)
                except TypeError:
                    legacy_layer = list(self.layer)
                    legacy_layer[0] = self.raw_input_num
                    self.net = NetClass(legacy_layer).float().to(device)
            else:
                self.net = NetClass(self.node_num, self.output_num).float().to(device)

            if self.load_state:
                load_path = (f"./Results/{self.ques_name}_{self.ini_num}/Models/"
                             f"{self.ques_name}_{self.ini_num}_"
                             f"{self.net.__module__.split('.')[-1]}.pth")
                self.net.load_state_dict(torch.load(load_path, map_location=device))

            # Create student network
            if self.bayesian_student_active:
                if self.student_type == 'mc_dropout':
                    import Module.Student_MCDropout as MCDrop
                    self.net_student = MCDrop.Net(
                        self.layer,
                        dropout_rate=self.dropout_rate,
                        heteroscedastic=self.heteroscedastic,
                        use_fourier_features=self.feature_enabled,
                        fourier_modes=self.fourier_modes,
                        hard_bc=self.use_hard_bc if self.is_poisson else False
                    ).float().to(device)
                elif self.student_type == 'vi_bnn':
                    import Module.Student_VIBNN as VIBNN
                    self.net_student = VIBNN.Net(
                        self.layer,
                        prior_sigma=self.prior_sigma,
                        heteroscedastic=self.heteroscedastic,
                        use_fourier_features=self.feature_enabled,
                        fourier_modes=self.fourier_modes,
                        hard_bc=self.use_hard_bc if self.is_poisson else False
                    ).float().to(device)
                else:
                    raise ValueError(f"Unknown student_type: {self.student_type}")
            elif self.distill_state:
                self.net_student = PINN.Net(
                    self.layer_student,
                    use_fourier_features=self.feature_enabled,
                    fourier_modes=self.fourier_modes,
                    hard_bc=self.use_hard_bc if self.is_poisson else False
                ).float().to(device)
            else:
                self.net_student = None

            print(f'\nRunning Model: {models_to_run[i]}\n')

            self.workflow()

            if len(models_to_run) > 1:
                group.loss_read(self.net.__module__.split('.')[-1])
                if self.monitor_state:
                    group.para_read(self.net.__module__.split('.')[-1])

        if len(models_to_run) > 1:
            group.loss_vis()
            if self.monitor_state:
                group.para_vis()

        if model_define_trigger == 0:
            raise ValueError(
                'The model name is incorrect. Please check again.')


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Scripts and Problems/ may reference these names from the v2.0 codebase
BayesianPsiNNTrainer = model


def burgers_residual(m, x, nu=0.01/np.pi):
    """Standalone Burgers residual (backward compat for Problems/*.py)."""
    x = x.clone().requires_grad_(True)
    u = m(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_s, u_t = u_x[:, 0:1], u_x[:, 1:2]
    u_xx = torch.autograd.grad(u_s, x, grad_outputs=torch.ones_like(u_s),
                                create_graph=True, retain_graph=True)[0][:, 0:1]
    return torch.mean((u_t + u * u_s - nu * u_xx) ** 2)


def laplace_residual(m, x):
    """Standalone Laplace residual (backward compat)."""
    x = x.clone().requires_grad_(True)
    u = m(x)
    ug = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u1, u2 = ug[:, 0:1], ug[:, 1:2]
    u11 = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1),
                               create_graph=True, retain_graph=True)[0][:, 0:1]
    u22 = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2),
                               create_graph=True, retain_graph=True)[0][:, 1:2]
    return torch.mean((u11 + u22) ** 2)


def poisson_residual(m, x):
    """Standalone Poisson residual (backward compat)."""
    return PT.poisson_residual_loss(m, x)

