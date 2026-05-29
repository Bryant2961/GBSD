# coding = utf-8
"""
Reconstruction trainer (Stage 3) for the structured candidate

Trains the compact StructuredPINN by optimizing only cluster centers
(and biases) while keeping the relation matrix R frozen.

Training losses include:
  1. PDE residual loss (autograd-based)
  2. Boundary/initial condition loss
  3. Optional data loss (supervised points)
  4. Optional distillation loss (teacher stabilization)

The PDE residual function is provided as a callback, making this
trainer agnostic to the specific equation being solved.

Third-party provenance for the migrated runtime engine is documented in
docs/THIRD_PARTY_NOTICE.md.
"""
import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from typing import Callable, Dict, Optional, Tuple, List


class ReconstructionTrainer:
    """
    Stage-3 reconstruction trainer for structured PINNs.

    Freezes the relation matrix R and trains only cluster centers + biases
    under physics-informed losses (PDE residual + boundary conditions).

    Args:
        structured_model: StructuredPINN instance (from Module.StructuredPINN)
        pde_residual_fn: Callable(model, x_collocation) -> scalar PDE loss.
            Must use autograd and accept the structured model as first arg.
        boundary_loss_fn: Callable(model) -> scalar BC/IC loss.
        teacher_model: Optional teacher for distillation stabilization.
        device: torch device.
        config: Dict of hyperparameters. Recognized keys:
            'lr': learning rate (default 1e-3)
            'epochs': number of training epochs (default 5000)
            'lr_step': step size for LR scheduler (default 2000)
            'lr_gamma': LR decay factor (default 0.5)
            'lambda_pde': weight for PDE residual loss (default 1.0)
            'lambda_bc': weight for BC/IC loss (default 1.0)
            'lambda_data': weight for data loss (default 1.0)
            'lambda_distill': weight for distillation loss (default 0.1)
            'grad_clip': gradient clipping value (default 1.0, None to disable)
            'print_every': print frequency (default 500)
            'seed': random seed (default 1234)
    """
    def __init__(self,
                 structured_model: nn.Module,
                 pde_residual_fn: Callable,
                 boundary_loss_fn: Callable,
                 teacher_model: Optional[nn.Module] = None,
                 anchor_model: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 config: Optional[Dict] = None):

        self.model = structured_model
        self.pde_residual_fn = pde_residual_fn
        self.boundary_loss_fn = boundary_loss_fn
        self.teacher = teacher_model
        self.anchor = anchor_model
        self.device = device or next(structured_model.parameters()).device

        # Default config
        cfg = {
            'lr': 1e-3,
            'epochs': 5000,
            'lr_step': 2000,
            'lr_gamma': 0.5,
            'lambda_pde': 1.0,
            'lambda_bc': 1.0,
            'lambda_data': 1.0,
            'lambda_distill': 0.1,
            'lambda_anchor': 0.0,
            'grad_clip': 1.0,
            'print_every': 500,
            'seed': 1234,
            'train_dropout_rate': None,
            'inference_dropout_rate': None,
            'anchor_pretrain_steps': 0,
            'anchor_pretrain_lr': None,
            'anchor_pretrain_pde_weight': 0.0,
            'residual_pretrain_steps': 0,
            'residual_pretrain_lr': None,
            'residual_pretrain_print_every': None,
            'lambda_residual_output': 0.0,
            'lambda_alpha': 0.0,
            'best_metric': 'total',
        }
        if config:
            cfg.update(config)
        self.config = cfg

        # Collocation and data points (set by user before training)
        self.x_collocation = None  # (N, input_dim) for PDE residual
        self.x_data = None         # (M, input_dim) supervised inputs
        self.u_data = None         # (M, output_dim) supervised targets
        self.x_validation = None   # optional held-out/reference points for best checkpoint selection
        self.u_validation = None

        # History
        self.history = {
            'loss': [], 'loss_pde': [], 'loss_bc': [],
            'loss_data': [], 'loss_distill': [], 'loss_anchor': [],
            'loss_residual': [], 'loss_alpha': [],
            'epoch': []
        }
        self.training_time = 0.0

    def set_collocation_points(self, x: torch.Tensor):
        """Set collocation points for PDE residual evaluation."""
        self.x_collocation = x.to(self.device).requires_grad_(True)

    def set_data_points(self, x: torch.Tensor, u: torch.Tensor):
        """Set supervised data points (optional)."""
        self.x_data = x.to(self.device)
        self.u_data = u.to(self.device)

    def set_validation_points(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        """Set validation points used only for best-checkpoint selection."""
        self.x_validation = x.to(self.device)
        self.u_validation = u.to(self.device) if u is not None else None

    def _compute_distillation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MSE distillation loss against teacher predictions."""
        if self.teacher is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            u_teacher = self.teacher(x)
            if isinstance(u_teacher, tuple):
                u_teacher = u_teacher[0]

        u_student = self.model(x)
        if isinstance(u_student, tuple):
            u_student = u_student[0]
        return F.mse_loss(u_student, u_teacher)

    def _compute_anchor_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Keep the structured model close to the dense pre-reconstruction model."""
        if self.anchor is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            u_anchor = self.anchor(x)
            if isinstance(u_anchor, tuple):
                u_anchor = u_anchor[0]

        u_student = self.model(x)
        if isinstance(u_student, tuple):
            u_student = u_student[0]
        return F.mse_loss(u_student, u_anchor)

    def _compute_data_loss(self) -> torch.Tensor:
        """Compute supervised data loss."""
        if self.x_data is None or self.u_data is None:
            return torch.tensor(0.0, device=self.device)

        u_pred = self.model(self.x_data)
        return F.mse_loss(u_pred, self.u_data)

    def _verify_gradient_flow(self):
        """Verify that centers receive gradients through R multiplication."""
        # Quick forward + backward check
        if self.x_collocation is None:
            return True

        self.model.train()
        x_test = self.x_collocation[:10].clone().requires_grad_(True)
        u = self.model(x_test)
        loss = u.sum()
        loss.backward()

        grad_ok = True
        for name, param in self.model.named_parameters():
            if 'centers' in name:
                if param.grad is None or param.grad.abs().sum() == 0:
                    print(f"  WARNING: No gradient flow to {name}")
                    grad_ok = False
                else:
                    print(f"  OK: gradient flows to {name} "
                          f"(grad norm={param.grad.norm().item():.6e})")

        self.model.zero_grad()
        return grad_ok

    def _selection_metric(self, total_loss: torch.Tensor) -> torch.Tensor:
        """Metric used to choose the checkpoint restored after reconstruction.

        Total physics loss can select a model with lower PDE residual but worse
        agreement with the dense Bayesian student.  For paper figures we need a
        structured model that remains faithful to the discovered dense solution,
        while PDE/BC terms still shape the optimization trajectory.
        """
        metric = str(self.config.get('best_metric', 'total')).strip().lower()
        if metric in ('total', 'loss'):
            return total_loss.detach()

        x_val = self.x_validation
        if x_val is None:
            x_val = self.x_collocation.detach()

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_val)
            if isinstance(pred, tuple):
                pred = pred[0]

            if metric in ('anchor', 'dense', 'student') and self.anchor is not None:
                target = self.anchor(x_val)
                if isinstance(target, tuple):
                    target = target[0]
                value = F.mse_loss(pred, target).detach()
            elif metric in ('teacher', 'distill') and self.teacher is not None:
                target = self.teacher(x_val)
                if isinstance(target, tuple):
                    target = target[0]
                value = F.mse_loss(pred, target).detach()
            elif metric in ('data', 'reference', 'exact') and self.u_validation is not None:
                value = F.mse_loss(pred, self.u_validation).detach()
            else:
                value = total_loss.detach()

        if was_training:
            self.model.train()
        return value

    def _residual_regularization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return residual-branch output and alpha penalties."""
        zero = torch.tensor(0.0, device=self.device)
        if getattr(self.model, 'residual_branch', None) is None:
            return zero, zero
        alpha = getattr(self.model, 'residual_alpha', None)
        if alpha is None:
            return zero, zero
        raw_x = x.detach()
        residual = alpha * self.model.residual_branch(raw_x)
        return torch.mean(residual ** 2), alpha.pow(2).mean()

    def _pretrain_structured_anchor(self, verbose: bool = True):
        """Warm start all structured trainable parameters against anchor/data."""
        steps = int(self.config.get('anchor_pretrain_steps', 0) or 0)
        if steps <= 0:
            return

        target_model = self.anchor if self.anchor is not None else self.teacher
        if target_model is None and (self.x_data is None or self.u_data is None):
            return

        lr = self.config.get('anchor_pretrain_lr', None)
        lr = float(lr) if lr is not None else float(self.config['lr'])
        print_every = max(1, steps // 5)
        pde_weight = float(self.config.get('anchor_pretrain_pde_weight', 0.0) or 0.0)
        lambda_res = float(self.config.get('lambda_residual_output', 0.0) or 0.0)
        lambda_alpha = float(self.config.get('lambda_alpha', 0.0) or 0.0)

        if target_model is not None:
            target_model.eval()
        self.model.train()

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return

        optimizer = optim.Adam(params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=lr * 0.05)

        if verbose:
            print(f"\n  Structured anchor warm start: {steps} steps, lr={lr:.2e}")

        best_metric = float('inf')
        best_state = None
        x = self.x_collocation.detach()

        for step in range(steps):
            optimizer.zero_grad()
            loss_anchor = torch.tensor(0.0, device=self.device)
            if target_model is not None:
                with torch.no_grad():
                    target = target_model(x)
                    if isinstance(target, tuple):
                        target = target[0]
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss_anchor = F.mse_loss(pred, target)

            loss_data = self._compute_data_loss()
            loss_pde = torch.tensor(0.0, device=self.device)
            if pde_weight > 0:
                loss_pde = self.pde_residual_fn(self.model, self.x_collocation)
            loss_residual, loss_alpha = self._residual_regularization(x)

            loss = (loss_anchor
                    + self.config['lambda_data'] * loss_data
                    + pde_weight * loss_pde
                    + lambda_res * loss_residual
                    + lambda_alpha * loss_alpha)
            loss.backward()
            if self.config['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(params, self.config['grad_clip'])
            optimizer.step()
            scheduler.step()

            select_value = float(self._selection_metric(loss).item())
            if select_value < best_metric:
                best_metric = select_value
                best_state = copy.deepcopy(self.model.state_dict())

            if verbose and ((step + 1) % print_every == 0 or step == steps - 1):
                alpha = getattr(self.model, 'residual_alpha', None)
                alpha_val = float(alpha.detach().cpu()) if alpha is not None else 0.0
                print(f"  Anchor warm start {step+1}/{steps} | "
                      f"Anchor: {loss_anchor.item():.4e} | "
                      f"Data: {loss_data.item():.4e} | "
                      f"Res: {loss_residual.item():.4e} | "
                      f"alpha={alpha_val:.4e}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _pretrain_residual_branch(self, verbose: bool = True):
        """Warm start the residual correction branch against anchor/teacher.

        The structured core is intentionally constrained by the discovered
        relation matrix.  A short residual-only warm start lets the correction
        branch absorb the dense model residual before the full physics loss is
        applied, which preserves accuracy when the discovered structure is
        slightly over-restrictive.
        """
        steps = int(self.config.get('residual_pretrain_steps', 0) or 0)
        if steps <= 0 or not hasattr(self.model, 'residual_branch'):
            return
        if getattr(self.model, 'residual_branch', None) is None:
            return

        target_model = self.anchor if self.anchor is not None else self.teacher
        if target_model is None:
            return

        lr = self.config.get('residual_pretrain_lr', None)
        lr = float(lr) if lr is not None else float(self.config['lr'])
        print_every = self.config.get('residual_pretrain_print_every', None)
        print_every = int(print_every) if print_every is not None else max(1, steps // 5)

        target_model.eval()
        self.model.train()

        old_requires_grad = {
            name: param.requires_grad
            for name, param in self.model.named_parameters()
        }
        for name, param in self.model.named_parameters():
            is_residual = ('residual_branch' in name or name == 'residual_alpha')
            param.requires_grad_(is_residual)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            for name, param in self.model.named_parameters():
                param.requires_grad_(old_requires_grad[name])
            return

        optimizer = optim.Adam(params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=lr * 0.05)

        if verbose:
            print(f"\n  Residual branch warm start: {steps} steps, lr={lr:.2e}")

        best_loss = float('inf')
        best_metric = float('inf')
        best_state = None
        x = self.x_collocation.detach()

        for step in range(steps):
            optimizer.zero_grad()
            with torch.no_grad():
                target = target_model(x)
                if isinstance(target, tuple):
                    target = target[0]
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss_residual, loss_alpha = self._residual_regularization(x)
            loss = (F.mse_loss(pred, target)
                    + float(self.config.get('lambda_residual_output', 0.0) or 0.0)
                    * loss_residual
                    + float(self.config.get('lambda_alpha', 0.0) or 0.0)
                    * loss_alpha)
            loss.backward()
            if self.config['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(params, self.config['grad_clip'])
            optimizer.step()
            scheduler.step()

            select_value = float(self._selection_metric(loss).item())
            if select_value < best_metric:
                best_metric = select_value
                best_loss = loss.item()
                best_state = copy.deepcopy(self.model.state_dict())

            if verbose and ((step + 1) % print_every == 0 or step == steps - 1):
                alpha = getattr(self.model, 'residual_alpha', None)
                alpha_val = float(alpha.detach().cpu()) if alpha is not None else 0.0
                print(f"  Residual warm start {step+1}/{steps} | "
                      f"MSE: {loss.item():.4e} | alpha={alpha_val:.4e}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        for name, param in self.model.named_parameters():
            param.requires_grad_(old_requires_grad[name])

    def train(self, verbose: bool = True,
              stage_label: str = 'Stage 4: Structured PINN Reconstruction Training') -> Dict:
        """
        Run reconstruction training.

        Args:
            verbose: Print training progress
            stage_label: Stage label for logging (default: Stage 4)

        Returns:
            Training history dictionary.
        """
        cfg = self.config
        torch.manual_seed(cfg['seed'])

        if verbose:
            print("\n" + "=" * 60)
            print(f"  {stage_label}")
            print("=" * 60)

            # Report compression
            if hasattr(self.model, 'count_parameters'):
                stats = self.model.count_parameters()
                print(f"  Trainable params:  {stats['trainable']}")
                print(f"  Original params:   {stats['original']}")
                print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

            # Verify gradient flow
            print("\n  Checking gradient flow through R @ centers...")
            self._verify_gradient_flow()

        assert self.x_collocation is not None, \
            "Must call set_collocation_points() before training."

        # Optimizer: only trainable parameters (centers + biases)
        optimizer = optim.Adam(self.model.parameters(), lr=cfg['lr'])

        # LR schedule: cosine annealing if requested, else step
        if cfg.get('lr_schedule', 'step') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01)
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg['lr_step'], gamma=cfg['lr_gamma'])

        # Distillation annealing schedule:
        #   Phase 1 (0 to warmup_frac): hold at lambda_d_start
        #     → structured model learns to faithfully reproduce student
        #   Phase 2 (warmup_frac to 1.0): cosine decay to lambda_d_final
        #     → gradually shift focus to physics (PDE + BC)
        lambda_d_start = cfg['lambda_distill']
        lambda_d_final = cfg.get('lambda_distill_final', lambda_d_start * 0.01)
        warmup_frac = cfg.get('distill_warmup_frac', 0.4)

        if self.teacher is not None:
            self.teacher.eval()
        if self.anchor is not None:
            self.anchor.eval()

        original_dropout_rate = cfg.get('inference_dropout_rate', None)
        train_dropout_rate = cfg.get('train_dropout_rate', None)
        if train_dropout_rate is not None and hasattr(self.model, 'set_dropout_rate'):
            self.model.set_dropout_rate(train_dropout_rate)
            if verbose:
                print(f"  Reconstruction dropout for training: {float(train_dropout_rate):.4f}")

        self.model.train()
        self._pretrain_structured_anchor(verbose=verbose)
        self._pretrain_residual_branch(verbose=verbose)
        start_time = time.time()
        best_loss = float('inf')
        best_metric = float('inf')
        best_state = None

        for epoch in range(cfg['epochs']):
            optimizer.zero_grad()

            # Annealed distillation weight
            frac = epoch / max(1, cfg['epochs'] - 1)
            if frac < warmup_frac:
                lambda_d_now = lambda_d_start  # hold
            else:
                decay_frac = (frac - warmup_frac) / (1.0 - warmup_frac)
                # cosine decay: smooth transition from start to final
                lambda_d_now = (lambda_d_final
                                + (lambda_d_start - lambda_d_final)
                                * 0.5 * (1.0 + math.cos(math.pi * decay_frac)))

            # PDE residual loss
            loss_pde = self.pde_residual_fn(self.model, self.x_collocation)

            # Boundary/IC loss
            loss_bc = self.boundary_loss_fn(self.model)

            # Data loss
            loss_data = self._compute_data_loss()

            # Distillation loss
            loss_distill = torch.tensor(0.0, device=self.device)
            if self.teacher is not None and lambda_d_now > 0:
                loss_distill = self._compute_distillation_loss(self.x_collocation)

            loss_anchor = torch.tensor(0.0, device=self.device)
            if self.anchor is not None and cfg.get('lambda_anchor', 0.0) > 0:
                loss_anchor = self._compute_anchor_loss(self.x_collocation)

            loss_residual, loss_alpha = self._residual_regularization(
                self.x_collocation)

            # Total loss
            loss = (cfg['lambda_pde'] * loss_pde +
                    cfg['lambda_bc'] * loss_bc +
                    cfg['lambda_data'] * loss_data +
                    lambda_d_now * loss_distill +
                    cfg.get('lambda_anchor', 0.0) * loss_anchor +
                    cfg.get('lambda_residual_output', 0.0) * loss_residual +
                    cfg.get('lambda_alpha', 0.0) * loss_alpha)

            loss.backward()

            # Gradient clipping
            if cfg['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg['grad_clip'])

            optimizer.step()
            scheduler.step()

            select_value = float(self._selection_metric(loss).item())
            if select_value < best_metric:
                best_metric = select_value
                best_loss = loss.item()
                best_state = copy.deepcopy(self.model.state_dict())

            # Record
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(loss.item())
            self.history['loss_pde'].append(loss_pde.item())
            self.history['loss_bc'].append(loss_bc.item())
            self.history['loss_data'].append(loss_data.item())
            self.history['loss_distill'].append(loss_distill.item())
            self.history['loss_anchor'].append(loss_anchor.item())
            self.history['loss_residual'].append(loss_residual.item())
            self.history['loss_alpha'].append(loss_alpha.item())

            # Also store on model for compatibility
            self.model.iter = epoch + 1
            self.model.iter_list.append(epoch + 1)
            self.model.loss_list.append(loss.item())
            self.model.loss_f_list.append(loss_pde.item())
            self.model.loss_b_list.append(loss_bc.item())

            if verbose and (epoch + 1) % cfg['print_every'] == 0:
                lr_now = scheduler.get_last_lr()[0]
                msg = (f"  Epoch {epoch+1}/{cfg['epochs']} | "
                       f"Loss: {loss.item():.4e} | "
                       f"PDE: {loss_pde.item():.4e} | "
                       f"BC: {loss_bc.item():.4e}")
                if self.x_data is not None:
                    msg += f" | Data: {loss_data.item():.4e}"
                if self.teacher is not None:
                    msg += f" | Distill: {loss_distill.item():.4e} (w={lambda_d_now:.3f})"
                if self.anchor is not None:
                    msg += f" | Anchor: {loss_anchor.item():.4e}"
                if cfg.get('lambda_residual_output', 0.0) > 0:
                    msg += f" | Res: {loss_residual.item():.4e}"
                msg += f" | LR: {lr_now:.2e}"
                print(msg)

        self.training_time = time.time() - start_time

        lbfgs_steps = int(cfg.get('lbfgs_steps', 0) or 0)
        if lbfgs_steps > 0:
            print(f"\n  L-BFGS reconstruction refinement: {lbfgs_steps} steps")
            optimizer_lbfgs = optim.LBFGS(
                self.model.parameters(),
                lr=cfg.get('lbfgs_lr', 1.0),
                max_iter=1,
                history_size=50,
                line_search_fn='strong_wolfe',
            )
            for step in range(lbfgs_steps):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss_pde = self.pde_residual_fn(self.model, self.x_collocation)
                    loss_bc = self.boundary_loss_fn(self.model)
                    loss_data = self._compute_data_loss()
                    loss_distill = torch.tensor(0.0, device=self.device)
                    if self.teacher is not None:
                        loss_distill = self._compute_distillation_loss(self.x_collocation)
                    loss_anchor = torch.tensor(0.0, device=self.device)
                    if self.anchor is not None and cfg.get('lambda_anchor', 0.0) > 0:
                        loss_anchor = self._compute_anchor_loss(self.x_collocation)
                    loss_residual, loss_alpha = self._residual_regularization(
                        self.x_collocation)
                    loss_total = (cfg['lambda_pde'] * loss_pde
                                  + cfg['lambda_bc'] * loss_bc
                                  + cfg['lambda_data'] * loss_data
                                  + cfg.get('lambda_distill_final',
                                            cfg['lambda_distill']) * loss_distill
                                  + cfg.get('lambda_anchor', 0.0) * loss_anchor
                                  + cfg.get('lambda_residual_output', 0.0)
                                  * loss_residual
                                  + cfg.get('lambda_alpha', 0.0) * loss_alpha)
                    loss_total.backward()
                    self._lbfgs_last = (loss_total, loss_pde, loss_bc,
                                        loss_data, loss_distill, loss_anchor,
                                        loss_residual, loss_alpha)
                    return loss_total

                optimizer_lbfgs.step(closure)
                (loss, loss_pde, loss_bc, loss_data,
                 loss_distill, loss_anchor,
                 loss_residual, loss_alpha) = self._lbfgs_last

                self.history['epoch'].append(cfg['epochs'] + step + 1)
                self.history['loss'].append(loss.item())
                self.history['loss_pde'].append(loss_pde.item())
                self.history['loss_bc'].append(loss_bc.item())
                self.history['loss_data'].append(loss_data.item())
                self.history['loss_distill'].append(loss_distill.item())
                self.history['loss_anchor'].append(loss_anchor.item())
                self.history['loss_residual'].append(loss_residual.item())
                self.history['loss_alpha'].append(loss_alpha.item())
                self.model.iter = cfg['epochs'] + step + 1
                self.model.iter_list.append(self.model.iter)
                self.model.loss_list.append(loss.item())
                self.model.loss_f_list.append(loss_pde.item())
                self.model.loss_b_list.append(loss_bc.item())

                select_value = float(self._selection_metric(loss).item())
                if select_value < best_metric:
                    best_metric = select_value
                    best_loss = loss.item()
                    best_state = copy.deepcopy(self.model.state_dict())

                if (step + 1) % cfg['print_every'] == 0 or step == lbfgs_steps - 1:
                    print(f"  L-BFGS {step+1}/{lbfgs_steps} | "
                          f"Loss: {loss.item():.4e} | "
                          f"PDE: {loss_pde.item():.4e} | "
                          f"BC: {loss_bc.item():.4e} | "
                          f"Distill: {loss_distill.item():.4e} | "
                          f"Anchor: {loss_anchor.item():.4e} | "
                          f"Res: {loss_residual.item():.4e}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if original_dropout_rate is not None and hasattr(self.model, 'set_dropout_rate'):
            self.model.set_dropout_rate(original_dropout_rate)
            if verbose and train_dropout_rate is not None:
                print(f"  Reconstruction dropout restored for MC inference: "
                      f"{float(original_dropout_rate):.4f}")
        self.model.eval()

        if verbose:
            print(f"\n  Reconstruction training complete. Time: {self.training_time:.2f}s")
            print(f"  Best loss: {best_loss:.6e}")
            print(f"  Best selection metric ({cfg.get('best_metric', 'total')}): "
                  f"{best_metric:.6e}")

        return self.history

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': self.config,
            'training_time': self.training_time,
            'model_class': self.model.__class__.__name__,
            'model_object': self.model,
        }
        for attr in ['use_fourier_features', 'fourier_modes', 'hard_bc',
                     'use_residual_branch']:
            if hasattr(self.model, attr):
                checkpoint[attr] = getattr(self.model, attr)
        if self.teacher is not None:
            checkpoint['teacher_state_dict'] = self.teacher.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.training_time = checkpoint.get('training_time', 0.0)


# =============================================================================
# Boundary loss factory functions (matching Training.py patterns)
# =============================================================================

def make_burgers_boundary_fn(x_min: float = -1.0, x_max: float = 1.0,
                              t_min: float = 0.0, t_max: float = 1.0,
                              n_points: int = 100,
                              ic_weight: float = 10.0,
                              ic_sign: float = -1.0,
                              device: torch.device = torch.device('cpu')) -> Callable:
    """Create boundary+IC loss function for Burgers equation.
    
    Args:
        ic_weight: Weight on initial condition loss. Use 10.0 for teacher training
                   (strong IC enforcement), 1.0 for reconstruction (IC already learned).
    """
    def boundary_loss(model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)

        t_b = torch.linspace(t_min, t_max, n_points, device=device).reshape(-1, 1)

        # Left BC: u(-1, t) = 0
        x_left = torch.full_like(t_b, x_min)
        u_left = model(torch.cat([x_left, t_b], dim=1))
        loss = loss + torch.mean(u_left ** 2)

        # Right BC: u(1, t) = 0
        x_right = torch.full_like(t_b, x_max)
        u_right = model(torch.cat([x_right, t_b], dim=1))
        loss = loss + torch.mean(u_right ** 2)

        # IC: u(x, 0) = -sin(pi*x)
        x_ic = torch.linspace(x_min, x_max, n_points, device=device).reshape(-1, 1)
        t_ic = torch.full_like(x_ic, t_min)
        u_ic = model(torch.cat([x_ic, t_ic], dim=1))
        u_ic_exact = ic_sign * torch.sin(np.pi * x_ic)
        loss = loss + ic_weight * torch.mean((u_ic - u_ic_exact) ** 2)

        return loss

    return boundary_loss


def make_laplace_boundary_fn(x_min: float = -1.0, x_max: float = 1.0,
                              y_min: float = -1.0, y_max: float = 1.0,
                              n_points: int = 100,
                              device: torch.device = torch.device('cpu')) -> Callable:
    """Create boundary loss for Laplace equation: u = x^3 - 3xy^2."""
    def boundary_loss(model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)

        # x = x_min boundary
        y_pts = torch.linspace(y_min, y_max, n_points, device=device).reshape(-1, 1)
        x_pts = torch.full_like(y_pts, x_min)
        u_exact = x_pts ** 3 - 3 * x_pts * y_pts ** 2
        u_pred = model(torch.cat([x_pts, y_pts], dim=1))
        loss = loss + torch.mean((u_pred - u_exact) ** 2)

        # x = x_max boundary
        x_pts = torch.full_like(y_pts, x_max)
        u_exact = x_pts ** 3 - 3 * x_pts * y_pts ** 2
        u_pred = model(torch.cat([x_pts, y_pts], dim=1))
        loss = loss + torch.mean((u_pred - u_exact) ** 2)

        # y = y_min boundary
        x_pts = torch.linspace(x_min, x_max, n_points, device=device).reshape(-1, 1)
        y_pts = torch.full_like(x_pts, y_min)
        u_exact = x_pts ** 3 - 3 * x_pts * y_pts ** 2
        u_pred = model(torch.cat([x_pts, y_pts], dim=1))
        loss = loss + torch.mean((u_pred - u_exact) ** 2)

        # y = y_max boundary
        y_pts = torch.full_like(x_pts, y_max)
        u_exact = x_pts ** 3 - 3 * x_pts * y_pts ** 2
        u_pred = model(torch.cat([x_pts, y_pts], dim=1))
        loss = loss + torch.mean((u_pred - u_exact) ** 2)

        return loss

    return boundary_loss


def make_poisson_boundary_fn(x_min: float = 0.0, x_max: float = 1.0,
                              y_min: float = 0.0, y_max: float = 1.0,
                              n_points: int = 100,
                              device: torch.device = torch.device('cpu')) -> Callable:
    """Create boundary loss for Poisson equation (zero Dirichlet)."""
    def boundary_loss(model: nn.Module) -> torch.Tensor:
        pts = []
        # All four boundaries
        y_b = torch.linspace(y_min, y_max, n_points, device=device).reshape(-1, 1)
        pts.append(torch.cat([torch.full_like(y_b, x_min), y_b], dim=1))
        pts.append(torch.cat([torch.full_like(y_b, x_max), y_b], dim=1))
        x_b = torch.linspace(x_min, x_max, n_points, device=device).reshape(-1, 1)
        pts.append(torch.cat([x_b, torch.full_like(x_b, y_min)], dim=1))
        pts.append(torch.cat([x_b, torch.full_like(x_b, y_max)], dim=1))

        x_all = torch.cat(pts, dim=0)
        u_pred = model(x_all)
        return torch.mean(u_pred ** 2)

    return boundary_loss


def get_boundary_fn(problem_name: str, config: Dict,
                    device: torch.device, ic_weight: float = 10.0) -> Callable:
    """Factory: get the right boundary loss function for a problem name.
    
    Args:
        ic_weight: IC weight for Burgers (10.0 for teacher, 1.0 for reconstruction).
    """
    if 'Burgers' in problem_name:
        return make_burgers_boundary_fn(
            x_min=config.get('x_min', -1.0), x_max=config.get('x_max', 1.0),
            t_min=config.get('y_min', 0.0), t_max=config.get('y_max', 1.0),
            n_points=int(config.get('bun_node_num', 100)),
            ic_weight=ic_weight,
            ic_sign=float(config.get('burgers_ic_sign', -1.0)),
            device=device
        )
    elif 'Laplace' in problem_name:
        return make_laplace_boundary_fn(
            x_min=config.get('x_min', -1.0), x_max=config.get('x_max', 1.0),
            y_min=config.get('y_min', -1.0), y_max=config.get('y_max', 1.0),
            n_points=int(config.get('bun_node_num', 100)), device=device
        )
    elif 'Poisson' in problem_name:
        return make_poisson_boundary_fn(
            x_min=config.get('x_min', 0.0), x_max=config.get('x_max', 1.0),
            y_min=config.get('y_min', 0.0), y_max=config.get('y_max', 1.0),
            n_points=int(config.get('bun_node_num', 100)), device=device
        )
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
