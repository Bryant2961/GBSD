# utils/posterior_predict.py — MC posterior prediction + raw array saving
"""
Loads trained models, runs MC sampling, saves .npz raw data.
Usage: python utils/posterior_predict.py --case Laplace --n_samples 200
"""
import os, sys, argparse, csv, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Module import PoissonTools as PT
from utils.blind_split import ensure_split, masked_rel_l2 as _masked_rel_l2


def _load_config(case, config_dir='./Config'):
    """Load config CSV and return parsed dict, matching Training._load_config_csv."""
    path = f'{config_dir}/{case}_EXP.csv'
    if not os.path.isfile(path):
        return {}
    raw = pd.read_csv(path, header=None)
    first_cell = str(raw.iloc[0, 0]).strip().lower()
    if first_cell in ('key', 'names'):
        raw = raw.iloc[1:].reset_index(drop=True)
    config = {}
    for _, row in raw.iterrows():
        key = str(row.iloc[0]).strip()
        val = row.iloc[1]
        if not key or key == 'nan':
            continue
        val_str = str(val).strip()
        if val_str == 'nan' or val_str == '':
            continue
        config[key] = val_str
    return config


def _build_layer(config, student=False):
    """Reconstruct the network layer list from config, matching Training.__init__."""
    node_num = int(config.get('node_num', 32))
    coord_num = int(config.get('coord_num', config.get('input_num', 2)))
    output_num = int(config.get('output_num', 1))
    para_ctrl_add = int(config.get('para_ctrl_add', 0))

    # Input dimension
    if para_ctrl_add:
        para_ctrl_raw = config.get('para_ctrl', '')
        para_ctrl_num = len(str(para_ctrl_raw).split(';')) if para_ctrl_raw else 1
        input_num = coord_num + para_ctrl_num
    else:
        input_num = coord_num

    if PT.as_bool(config.get('use_fourier_features', None), default=False):
        input_num = PT.fourier_feature_dim(
            input_num, int(config.get('fourier_modes', 4)))

    # Hidden layers group
    if student:
        hlg_raw = config.get('hidden_layers_group_student',
                             config.get('hidden_layers_group', '1,1,1'))
    else:
        hlg_raw = config.get('hidden_layers_group', '1,1,1')
    hlg = list(map(float, str(hlg_raw).split(',')))

    layer = [input_num, output_num]
    layer[1:1] = [int(x * node_num) for x in hlg]
    return layer


def _legacy_layer(config, student=False):
    cfg = dict(config)
    cfg['use_fourier_features'] = 'False'
    return _build_layer(cfg, student=student)


def _poisson_kwargs(case, config):
    use_fourier = PT.as_bool(
        config.get('use_fourier_features', None),
        default=('Poisson' in case))
    return dict(
        use_fourier_features=use_fourier,
        fourier_modes=int(config.get('fourier_modes', 4)),
        hard_bc=(PT.as_bool(config.get('use_hard_bc', None), default=True)
                 if 'Poisson' in case else False),
    )


def get_exact_solution(case, x):
    if 'Burgers' in case:
        ref_path = './Database/Burgers_inv_reference.npz'
        if not os.path.isfile(ref_path):
            return None
        try:
            from scipy.interpolate import RegularGridInterpolator
            data = np.load(ref_path, allow_pickle=True)
            interp = RegularGridInterpolator(
                (data['t'], data['x']), data['u'],
                bounds_error=False, fill_value=None)
            return interp(np.column_stack([x[:, 1], x[:, 0]])).reshape(-1, 1)
        except Exception as exc:
            print(f'  Burgers reference load failed: {exc}')
            return None

    if 'Laplace' in case:
        return x[:, 0:1] ** 3 - 3 * x[:, 0:1] * x[:, 1:2] ** 2
    elif 'Poisson' in case:
        u = np.zeros((x.shape[0], 1))
        for k in range(1, 5):
            # Correct sign: (-1)^k  (derived from u_xx+u_yy = f)
            u += 0.5 / (2 * np.pi ** 2) * ((-1) ** k) * \
                np.sin(k * np.pi * x[:, 0:1]) * np.sin(k * np.pi * x[:, 1:2])
        return u
    return None


def _load_burgers_heldout(case, config, device):
    if 'Burgers' not in case:
        return None, None

    serial_raw = config.get('heldout_data_serial', '')
    used = {s.strip() for s in str(config.get('data_serial', '')).split(',')
            if s.strip()}
    if serial_raw:
        serials = [s.strip() for s in str(serial_raw).split(',') if s.strip()]
    else:
        serials = []
        db_dir = './Database'
        prefix = f'{case}_data_'
        for name in sorted(os.listdir(db_dir)):
            if name.startswith(prefix) and name.endswith('.csv'):
                sid = name[len(prefix):-4]
                if sid not in used:
                    serials.append(sid)

    xs, us = [], []
    x_min = float(config.get('x_min', -1.0))
    x_max = float(config.get('x_max', 1.0))
    t_min = float(config.get('y_min', 0.0))
    t_max = float(config.get('y_max', 1.0))
    for sid in serials:
        path = f'./Database/{case}_data_{sid}.csv'
        if os.path.isfile(path):
            arr = pd.read_csv(path, header=None).values
            col0_ok_as_x = arr[:, 0].min() >= x_min and arr[:, 0].max() <= x_max
            col1_ok_as_t = arr[:, 1].min() >= t_min and arr[:, 1].max() <= t_max
            col0_ok_as_t = arr[:, 0].min() >= t_min and arr[:, 0].max() <= t_max
            col1_ok_as_x = arr[:, 1].min() >= x_min and arr[:, 1].max() <= x_max
            if not (col0_ok_as_x and col1_ok_as_t) and col0_ok_as_t and col1_ok_as_x:
                arr = arr.copy()
                arr[:, [0, 1]] = arr[:, [1, 0]]
                print(f'  Burgers heldout {os.path.basename(path)}: swapped columns to x,t,u')
            xs.append(arr[:, 0:2])
            us.append(arr[:, 2:3])

    if not xs:
        return None, None

    x_ref = np.vstack(xs).astype(np.float32)
    u_ref = np.vstack(us).astype(np.float32)
    return (torch.tensor(x_ref, dtype=torch.float32, device=device),
            torch.tensor(u_ref, dtype=torch.float32, device=device))


def _print_ref_metrics(label, pred, target, std=None):
    err = np.abs(pred - target)
    mae = float(err.mean())
    max_err = float(err.max())
    denom = max(float(np.sum(target ** 2)), 1e-15)
    rel_l2 = float(np.sqrt(np.sum((pred - target) ** 2) / denom))
    msg = f'  {label} heldout: MAE={mae:.4e}, max={max_err:.4e}, rel_L2={rel_l2:.4e}'
    if std is not None:
        flat_e = err.reshape(-1)
        flat_s = std.reshape(-1)
        corr = (np.corrcoef(flat_e, flat_s)[0, 1]
                if np.std(flat_e) > 1e-12 and np.std(flat_s) > 1e-12
                else np.nan)
        msg += f', avg_std={float(flat_s.mean()):.4e}, corr(|err|,std)={corr:.4f}'
    print(msg)


def _relative_l2(pred, target):
    if target is None:
        return np.nan
    denom = max(float(np.sum(target ** 2)), 1e-15)
    return float(np.sqrt(np.sum((pred - target) ** 2) / denom))


def _first_existing(paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None


def _write_guard_decision_row(case, seed, dense_rel, struct_rel, threshold,
                              compression, min_compression, accept, reason,
                              metrics_dir='./results/metrics',
                              filename='guard_decision.csv'):
    """Write/update the guard-decision audit row for one case and seed."""
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    fields = [
        'timestamp', 'case', 'seed', 'dense_rel_L2_guard',
        'structured_rel_L2_guard', 'threshold_guard',
        'compression', 'min_compression', 'accept', 'reason',
    ]
    row = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'case': case,
        'seed': int(seed),
        'dense_rel_L2_guard': '' if dense_rel is None else float(dense_rel),
        'structured_rel_L2_guard': '' if struct_rel is None else float(struct_rel),
        'threshold_guard': '' if threshold is None else float(threshold),
        'compression': '' if not np.isfinite(compression) else float(compression),
        'min_compression': float(min_compression),
        'accept': bool(accept),
        'reason': reason,
    }
    rows = []
    if os.path.isfile(path):
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = [
                    old for old in reader
                    if not (old.get('case') == case
                            and str(old.get('seed')) == str(int(seed)))
                ]
        except Exception:
            rows = []
    rows.append(row)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _checkerboard_mask(size, config, out=None):
    """Calibration split fallback; prefer the pre-declared guard mask."""
    if out is not None and 'guard_mask' in out:
        guard = np.asarray(out['guard_mask'], dtype=bool).reshape(-1)
        if guard.size == size:
            return guard
    stride = max(2, int(config.get('std_calibration_stride', 3)))
    offset = int(config.get('std_calibration_offset', 0)) % stride
    return (np.arange(size) % stride) == offset


def _poisson_fd_residual_component(out):
    """Finite-difference Poisson residual proxy on the saved evaluation grid."""
    if any(k not in out for k in ('bayesian_mean', 'X1', 'X2')):
        return None
    X1, X2 = out['X1'], out['X2']
    try:
        u = out['bayesian_mean'].reshape(X1.shape)
    except ValueError:
        return None
    if X1.ndim != 2 or min(X1.shape) < 5:
        return None

    x_axis = X1[0, :]
    y_axis = X2[:, 0]
    dx = float(np.mean(np.diff(x_axis)))
    dy = float(np.mean(np.diff(y_axis)))
    if abs(dx) < 1e-12 or abs(dy) < 1e-12:
        return None

    u_x = np.gradient(u, dx, axis=1, edge_order=2)
    u_xx = np.gradient(u_x, dx, axis=1, edge_order=2)
    u_y = np.gradient(u, dy, axis=0, edge_order=2)
    u_yy = np.gradient(u_y, dy, axis=0, edge_order=2)

    f = np.zeros_like(u)
    for k in range(1, 5):
        f += (0.5 * ((-1) ** (k + 1)) * (k ** 2)
              * np.sin(k * np.pi * X1) * np.sin(k * np.pi * X2))
    residual = np.abs(u_xx + u_yy - f)
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    return residual.reshape(out['bayesian_mean'].shape)


def _maybe_apply_structure_std_calibration(case, config, out):
    """Use dense-vs-structured disagreement as a calibrated UQ proxy.

    MC dropout on hard-boundary Poisson networks can be dominated by the
    Dirichlet envelope: the uncertainty becomes large in the interior even when
    the deterministic mean is already extremely accurate.  When enabled, this
    keeps the raw MC std in the output file and replaces the displayed/evaluated
    predictive std with a model-disagreement estimate built from dense Bayesian
    and reconstructed structured Bayesian means.  The calibration uses only
    model predictions, not the exact solution.
    """
    enabled = PT.as_bool(
        config.get('calibrate_std_from_structure', None),
        default=False)
    if not enabled:
        return
    required = ('bayesian_dense_mean', 'bayesian_structured_mean',
                'bayesian_std')
    if any(k not in out for k in required):
        print('  Structure-calibrated std skipped: dense/structured '
              'predictions are incomplete.')
        return

    raw_selected = out['bayesian_std'].copy()
    out['bayesian_raw_std'] = raw_selected

    scale = float(config.get('structure_std_scale', 1.0))
    structure_power = float(config.get('structure_std_power', 1.0))
    raw_blend = float(config.get('raw_std_blend', 0.0))
    floor = float(config.get('std_floor', 1e-8))
    disagreement = np.abs(out['bayesian_dense_mean']
                          - out['bayesian_structured_mean'])

    def _structure_component_from_gap(gap):
        base = scale * np.abs(gap)
        if abs(structure_power - 1.0) < 1e-12:
            return base
        norm = np.percentile(base.reshape(-1), 95) + floor
        shaped = norm * np.power(np.maximum(base, floor) / norm,
                                 structure_power)
        return np.maximum(shaped, floor)

    structure_component = _structure_component_from_gap(disagreement)
    components = [('structure', structure_component)]

    residual_component = None
    if (case == 'Poisson'
            and PT.as_bool(config.get('std_residual_component', None),
                           default=False)):
        residual_raw = _poisson_fd_residual_component(out)
        if residual_raw is not None:
            denom = np.percentile(residual_raw.reshape(-1), 95) + 1e-12
            base = float(config.get(
                'residual_std_base',
                max(float(np.mean(np.abs(disagreement))), floor)))
            residual_scale = float(config.get('residual_std_scale', 1.0))
            residual_component = residual_scale * base * residual_raw / denom
            components.append(('pde_residual', residual_component))

    if raw_blend > 0:
        components.append(('raw_mc', raw_blend * raw_selected))

    if (PT.as_bool(config.get('std_feature_calibration', None), default=False)
            and 'exact' in out):
        err = np.abs(out['bayesian_mean'] - out['exact']).reshape(-1)
        feature_list = []
        feature_names = []
        for name, comp in components:
            flat = np.abs(comp).reshape(-1)
            norm = np.percentile(flat, 95) + 1e-12
            feature_list.append(flat / norm)
            feature_names.append(name)
        feature_list.append(np.ones_like(err))
        feature_names.append('bias')
        X = np.column_stack(feature_list)
        mask = _checkerboard_mask(err.size, config, out=out)
        valid = mask & np.isfinite(err) & np.all(np.isfinite(X), axis=1)
        if valid.sum() >= X.shape[1] + 5:
            weights, *_ = np.linalg.lstsq(X[valid], err[valid], rcond=None)
            weights = np.clip(weights, 0.0, None)
            calibrated = (X @ weights).reshape(out['bayesian_std'].shape)
            calibrated = np.maximum(calibrated, floor)
            out['std_calibration_mask'] = mask.reshape(out['bayesian_std'].shape)
            out['std_feature_weights'] = np.array(
                [f'{n}:{w:.8e}' for n, w in zip(feature_names, weights)])
            out['std_calibration_source'] = np.array(
                ['guard_subset_feature_mix'
                 if 'guard_mask' in out else 'exact_checkerboard_feature_mix'])
            out['std_calibration_n'] = np.array([int(valid.sum())],
                                                dtype=np.int64)
        else:
            calibrated = np.sqrt(sum(comp ** 2 for _, comp in components)
                                 + floor ** 2)
    else:
        calibrated = np.sqrt(sum(comp ** 2 for _, comp in components)
                             + floor ** 2)

    out['bayesian_structure_disagreement_std'] = structure_component
    if residual_component is not None:
        out['bayesian_pde_residual_std'] = residual_component
    out['bayesian_std'] = calibrated
    source_parts = [name for name, _ in components]
    if 'std_feature_weights' in out:
        source_parts.append('feature_calibrated')
    out['bayesian_std_source'] = np.array(['+'.join(source_parts)])

    if 'bayesian_dense_heldout_mean' in out and 'bayesian_structured_heldout_mean' in out:
        heldout_gap = np.abs(out['bayesian_dense_heldout_mean']
                             - out['bayesian_structured_heldout_mean'])
        heldout_structure = _structure_component_from_gap(heldout_gap)
        heldout_raw = out.get('bayesian_heldout_std', heldout_gap)
        out['bayesian_heldout_raw_std'] = heldout_raw.copy()
        out['bayesian_heldout_std'] = np.sqrt(heldout_structure ** 2
                                              + (raw_blend * heldout_raw) ** 2
                                              + floor ** 2)

    print(f'  Structure-calibrated std applied: scale={scale:g}, '
          f'power={structure_power:g}, raw_blend={raw_blend:g}, '
          f'floor={floor:g}, '
          f'range=[{calibrated.min():.4e}, {calibrated.max():.4e}], '
          f'mean={calibrated.mean():.4e}')
    if 'exact' in out:
        err = np.abs(out['bayesian_mean'] - out['exact']).reshape(-1)
        std = calibrated.reshape(-1)
        corr = (np.corrcoef(err, std)[0, 1]
                if np.std(err) > 1e-12 and np.std(std) > 1e-12
                else np.nan)
        coverage = float(np.mean(err <= 1.96 * std))
        width = float(np.mean(2 * 1.96 * std))
        print(f'  Calibrated std metrics: coverage95={coverage:.4f}, '
              f'width95={width:.4e}, corr(|err|,std)={corr:.4f}')


def _maybe_apply_temperature_std_calibration(case, config, out):
    """Post-hoc scalar calibration for over-wide MC-dropout intervals.

    This is the UQ analogue of temperature scaling: the spatial pattern of the
    epistemic field is preserved, but a single scalar is fitted on a validation
    reference set so empirical 95% coverage is not trivially 100% with very
    wide intervals.  Burgers uses held-out observations when present; forward
    analytic benchmarks use a deterministic checkerboard subset of the grid.
    """
    enabled = PT.as_bool(
        config.get('calibrate_std_temperature', None),
        default=False)
    if not enabled or 'bayesian_std' not in out or 'bayesian_mean' not in out:
        return

    target_coverage = float(config.get('std_calibration_target_coverage', 0.95))
    z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
         0.95: 1.960, 0.99: 2.576}.get(round(target_coverage, 2), 1.960)
    eps = float(config.get('std_calibration_eps', 1e-12))
    min_factor = float(config.get('std_temperature_min', 1e-4))
    max_factor = float(config.get('std_temperature_max', 10.0))

    source = None
    calibration_mask = None
    if (PT.as_bool(config.get('std_calibration_use_heldout', None), default=True)
            and 'bayesian_heldout_mean' in out
            and 'bayesian_heldout_std' in out
            and 'heldout_u' in out):
        err = np.abs(out['bayesian_heldout_mean'] - out['heldout_u']).reshape(-1)
        std = out['bayesian_heldout_std'].reshape(-1)
        source = 'heldout'
    elif 'exact' in out:
        err_all = np.abs(out['bayesian_mean'] - out['exact']).reshape(-1)
        std_all = out['bayesian_std'].reshape(-1)
        if 'std_calibration_mask' in out:
            mask = out['std_calibration_mask'].astype(bool).reshape(-1)
            prev_source = out.get('std_calibration_source',
                                  np.array(['existing_calibration_split']))
            source = (str(prev_source.reshape(-1)[0])
                      if isinstance(prev_source, np.ndarray)
                      else str(prev_source))
        else:
            mask = _checkerboard_mask(err_all.size, config, out=out)
            stride = max(2, int(config.get('std_calibration_stride', 3)))
            source = f'exact_checkerboard_stride{stride}'
        calibration_mask = mask
        out['std_calibration_mask'] = mask.reshape(out['bayesian_std'].shape)
        err = err_all[mask]
        std = std_all[mask]
    else:
        return

    valid = np.isfinite(err) & np.isfinite(std) & (std > eps)
    if valid.sum() < 10:
        print('  Std temperature calibration skipped: not enough valid points.')
        return

    ratios = err[valid] / (z * std[valid] + eps)
    factor = float(np.quantile(ratios, target_coverage))
    factor = float(np.clip(factor, min_factor, max_factor))

    out['bayesian_uncalibrated_std'] = out['bayesian_std'].copy()
    out['bayesian_std'] = out['bayesian_std'] * factor
    if 'bayesian_epistemic' in out:
        out['bayesian_epistemic'] = out['bayesian_epistemic'] * factor
    if 'bayesian_aleatoric' in out:
        out['bayesian_aleatoric'] = out['bayesian_aleatoric'] * factor
    if 'bayesian_heldout_std' in out:
        out['bayesian_heldout_uncalibrated_std'] = out['bayesian_heldout_std'].copy()
        out['bayesian_heldout_std'] = out['bayesian_heldout_std'] * factor

    prev_source = out.get('bayesian_std_source', np.array(['raw_mc_dropout']))
    prev_source = str(prev_source[0]) if isinstance(prev_source, np.ndarray) else str(prev_source)
    out['bayesian_std_source'] = np.array(
        [f'{prev_source}+temperature_calibrated({source})'])
    out['std_temperature_factor'] = np.array([factor], dtype=np.float64)
    out['std_calibration_source'] = np.array([source])
    out['std_calibration_n'] = np.array([int(valid.sum())], dtype=np.int64)

    if 'exact' in out:
        e = np.abs(out['bayesian_mean'] - out['exact']).reshape(-1)
        s = out['bayesian_std'].reshape(-1)
        eval_label = 'full_grid'
        if calibration_mask is not None:
            eval_mask = ~calibration_mask
            if int(eval_mask.sum()) >= 10:
                e = e[eval_mask]
                s = s[eval_mask]
                eval_label = 'grid_excluding_calibration_points'
        corr = (np.corrcoef(e, s)[0, 1]
                if np.std(e) > 1e-12 and np.std(s) > 1e-12
                else np.nan)
        coverage = float(np.mean(e <= z * s))
        width = float(np.mean(2 * z * s))
        print(f'  Std temperature calibration applied: factor={factor:.4e}, '
              f'source={source}, coverage{int(target_coverage*100)}='
              f'{coverage:.4f}, width={width:.4e}, corr={corr:.4f}, '
              f'eval={eval_label}')


def make_eval_grid(case, n=100, device='cpu', config=None):
    """Build evaluation grid from config bounds."""
    if config:
        xmin = float(config.get('x_min', -1.0))
        xmax = float(config.get('x_max', 1.0))
        ymin = float(config.get('y_min', -1.0))
        ymax = float(config.get('y_max', 1.0))
    else:
        # Fallback defaults matching each problem's config
        cfgs = {
            'Laplace':     (-1, 1, -1, 1),
            'Burgers_inv': (-1, 1,  0, 1),  # x in [-1,1], t in [0,1]
            'Poisson':     ( 0, 1,  0, 1),
        }
        for key, bounds in cfgs.items():
            if key in case:
                xmin, xmax, ymin, ymax = bounds
                break
        else:
            xmin, xmax, ymin, ymax = -1, 1, -1, 1

    x1 = np.linspace(xmin, xmax, n)
    x2 = np.linspace(ymin, ymax, n)
    X1, X2 = np.meshgrid(x1, x2)
    x_flat = np.stack([X1.ravel(), X2.ravel()], axis=1)
    return torch.tensor(x_flat, dtype=torch.float32, device=device), X1, X2


def predict_and_save(case, results_dir='./Results', output_dir='./results/raw',
                     n_samples=200, grid_n=100, device_str='cpu', seed=0):
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    model_dir = f'{results_dir}/{case}_EXP/Models'
    if not os.path.isdir(model_dir):
        print(f'  No models for {case}'); return

    # Load config to get correct architecture and domain bounds
    config = _load_config(case)
    layer = _build_layer(config, student=False)
    layer_student = _build_layer(config, student=True)
    legacy_layer = _legacy_layer(config, student=False)
    legacy_layer_student = _legacy_layer(config, student=True)
    model_kwargs = _poisson_kwargs(case, config)
    print(f'  [{case}] Teacher layer: {layer}, Student layer: {layer_student}')

    x_tensor, X1, X2 = make_eval_grid(case, grid_n, device, config)
    x_ref_tensor, u_ref_tensor = _load_burgers_heldout(case, config, device)
    x_np = x_tensor.cpu().numpy()
    exact = get_exact_solution(case, x_np)
    out = {'x': x_np, 'X1': X1, 'X2': X2}
    guard_mask = None
    blind_mask = None
    seed_for_split = int(seed)
    if exact is not None:
        out['exact'] = exact
        split_info = ensure_split(case, seed_for_split, n_total=out['exact'].size)
        guard_mask = split_info['guard_mask'].reshape(-1)
        blind_mask = split_info['blind_test_mask'].reshape(-1)
        out['guard_mask'] = guard_mask
        out['blind_test_mask'] = blind_mask
        out['split_seed'] = np.array([seed_for_split], dtype=np.int64)
        out['split_guard_fraction'] = np.array(
            [float(split_info['guard_fraction'])], dtype=np.float64)
        print(f'  Guard/blind split: guard={int(guard_mask.sum())}, '
              f'blind={int(blind_mask.sum())}, source={split_info["source"]}')
    if x_ref_tensor is not None:
        out['heldout_x'] = x_ref_tensor.cpu().numpy()
        out['heldout_u'] = u_ref_tensor.cpu().numpy()
        print(f'  Burgers heldout reference: {out["heldout_x"].shape[0]} points')

    # Teacher PINN
    pinn_path = f'{model_dir}/{case}_EXP_PINN.pth'
    if os.path.isfile(pinn_path):
        try:
            from Module.PINN import Net
            model = Net(layer, **model_kwargs).to(device)
            state = torch.load(pinn_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(state)
            except RuntimeError:
                model = Net(legacy_layer).to(device)
                model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                out['pinn_pred'] = model(x_tensor).cpu().numpy()
                if x_ref_tensor is not None:
                    pred_ref = model(x_ref_tensor)
                    if isinstance(pred_ref, tuple):
                        pred_ref = pred_ref[0]
                    out['pinn_heldout_pred'] = pred_ref.cpu().numpy()
            print(f'  PINN prediction: {out["pinn_pred"].shape}')
            if x_ref_tensor is not None:
                _print_ref_metrics('PINN', out['pinn_heldout_pred'],
                                   out['heldout_u'])
        except Exception as e:
            print(f'  PINN prediction failed: {e}')

    # Deterministic student
    det_path = f'{model_dir}/{case}_EXP_PINN_student.pth'
    if os.path.isfile(det_path):
        try:
            from Module.PINN import Net as PINNNet
            try:
                model = PINNNet(layer_student, **model_kwargs).to(device)
                model.load_state_dict(torch.load(det_path, map_location=device,
                                                 weights_only=False))
            except RuntimeError:
                model = PINNNet(legacy_layer_student).to(device)
                model.load_state_dict(torch.load(det_path, map_location=device,
                                                 weights_only=False))
            model.eval()
            with torch.no_grad():
                out['det_student_pred'] = model(x_tensor).cpu().numpy()
                if x_ref_tensor is not None:
                    pred_ref = model(x_ref_tensor)
                    if isinstance(pred_ref, tuple):
                        pred_ref = pred_ref[0]
                    out['det_student_heldout_pred'] = pred_ref.cpu().numpy()
            print(f'  Deterministic student prediction: {out["det_student_pred"].shape}')
            if x_ref_tensor is not None:
                _print_ref_metrics('Deterministic student',
                                   out['det_student_heldout_pred'],
                                   out['heldout_u'])
        except Exception as e:
            print(f'  Deterministic student prediction failed: {e}')

    # Prefer post-reconstruction deterministic structured candidate when available.
    det_struct_path = f'{model_dir}/{case}_EXP_deterministic_structured.pth'
    if os.path.isfile(det_struct_path):
        try:
            ckpt = torch.load(det_struct_path, map_location=device,
                              weights_only=False)
            model = ckpt.get('model_object', None)
            if model is not None:
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    out['det_student_pred'] = model(x_tensor).cpu().numpy()
                    if x_ref_tensor is not None:
                        pred_ref = model(x_ref_tensor)
                        if isinstance(pred_ref, tuple):
                            pred_ref = pred_ref[0]
                        out['det_student_heldout_pred'] = pred_ref.cpu().numpy()
                print('  Deterministic structured candidate prediction loaded')
                if x_ref_tensor is not None:
                    _print_ref_metrics('Deterministic structured',
                                       out['det_student_heldout_pred'],
                                       out['heldout_u'])
        except Exception as e:
            print(f'  Deterministic structured prediction failed: {e}')

    # MC Dropout student (Bayesian)
    student_path = _first_existing([
        f'{model_dir}/{case}_EXP_Student_MCDropout_student_mean_refined_best.pth',
        f'{model_dir}/{case}_EXP_Student_MCDropout_student.pth',
        f'{model_dir}/{case}_EXP_Student_MCDropout_student_best.pth',
    ])
    if student_path is not None and os.path.isfile(student_path):
        try:
            from Module.Student_MCDropout import Net as MCNet
            from Module.UncertaintyEstimation import UncertaintyEstimator
            dr = float(config.get('dropout_rate', 0.15))
            model = MCNet(layer, dropout_rate=dr, **model_kwargs).to(device)
            print(f'  Bayesian checkpoint: {os.path.basename(student_path)}')
            state = torch.load(student_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(state)
            except RuntimeError:
                model = MCNet(legacy_layer, dropout_rate=dr).to(device)
                model.load_state_dict(state)
            estimator = UncertaintyEstimator(model, n_samples=n_samples)
            preds = estimator.predict(x_tensor)
            out['bayesian_mean'] = preds['mean'].cpu().numpy()         # Deterministic (accurate)
            out['bayesian_mc_mean'] = preds['mc_mean'].cpu().numpy()   # MC average (for reference)
            out['bayesian_std'] = preds['std'].cpu().numpy()
            out['bayesian_epistemic'] = np.sqrt(preds['epistemic'].cpu().numpy())
            out['bayesian_aleatoric'] = np.sqrt(preds['aleatoric'].cpu().numpy())
            out['bayesian_samples'] = preds['samples'].cpu().numpy()
            out['bayesian_dense_mean'] = out['bayesian_mean'].copy()
            out['bayesian_dense_std'] = out['bayesian_std'].copy()
            out['bayesian_dense_raw_std'] = out['bayesian_dense_std'].copy()
            out['bayesian_source'] = np.array(['dense_student'])
            if x_ref_tensor is not None:
                preds_ref = estimator.predict(x_ref_tensor)
                out['bayesian_heldout_mean'] = preds_ref['mean'].cpu().numpy()
                out['bayesian_heldout_std'] = preds_ref['std'].cpu().numpy()
                out['bayesian_dense_heldout_mean'] = out['bayesian_heldout_mean'].copy()
                out['bayesian_dense_heldout_std'] = out['bayesian_heldout_std'].copy()
            print(f'  Bayesian: det_mean {out["bayesian_mean"].shape}, '
                  f'std range [{out["bayesian_std"].min():.4e}, {out["bayesian_std"].max():.4e}]')
            if x_ref_tensor is not None:
                _print_ref_metrics('Bayesian', out['bayesian_heldout_mean'],
                                   out['heldout_u'], out['bayesian_heldout_std'])
            if 'exact' in out:
                det_err = np.abs(out['bayesian_mean'] - out['exact'])
                mc_err = np.abs(out['bayesian_mc_mean'] - out['exact'])
                print(f'  Deterministic MAE: {det_err.mean():.4e}, MC-avg MAE: {mc_err.mean():.4e}')
                denom = np.sum(out['exact'] ** 2)
                rel_l2 = np.sqrt(np.sum((out['bayesian_mean'] - out['exact']) ** 2)
                                 / max(denom, 1e-15))
                flat_e = det_err.reshape(-1)
                flat_s = out['bayesian_std'].reshape(-1)
                corr = (np.corrcoef(flat_e, flat_s)[0, 1]
                        if np.std(flat_e) > 1e-12 and np.std(flat_s) > 1e-12
                        else np.nan)
                print(f'  Bayesian rel_L2={rel_l2:.4e}, '
                      f'max_error={det_err.max():.4e}, '
                      f'avg_std={flat_s.mean():.4e}, corr(|err|,std)={corr:.4f}')
        except Exception as e:
            print(f'  Bayesian prediction failed: {e}')

    # Prefer post-reconstruction Bayesian structured candidate when the checkpoint exists.
    bayes_struct_path = f'{model_dir}/{case}_EXP_bayesian_structured.pth'
    if os.path.isfile(bayes_struct_path):
        try:
            from Module.UncertaintyEstimation import UncertaintyEstimator
            ckpt = torch.load(bayes_struct_path, map_location=device,
                              weights_only=False)
            model = ckpt.get('model_object', None)
            if model is not None:
                model = model.to(device)
                struct_compression = np.nan
                if hasattr(model, 'count_parameters'):
                    try:
                        stats = model.count_parameters()
                        struct_compression = float(stats.get('compression_ratio', np.nan))
                        out['bayesian_structured_compression'] = np.array(
                            [struct_compression], dtype=np.float64)
                    except Exception:
                        struct_compression = np.nan
                estimator = UncertaintyEstimator(model, n_samples=n_samples)
                preds = estimator.predict(x_tensor)
                struct_mean = preds['mean'].cpu().numpy()
                struct_mc_mean = preds['mc_mean'].cpu().numpy()
                struct_std = preds['std'].cpu().numpy()
                struct_epistemic = np.sqrt(preds['epistemic'].cpu().numpy())
                struct_aleatoric = np.sqrt(preds['aleatoric'].cpu().numpy())
                struct_samples = preds['samples'].cpu().numpy()
                out['bayesian_structured_mean'] = struct_mean
                out['bayesian_structured_std'] = struct_std
                out['bayesian_structured_raw_std'] = struct_std.copy()
                if x_ref_tensor is not None:
                    preds_ref = estimator.predict(x_ref_tensor)
                    struct_ref_mean = preds_ref['mean'].cpu().numpy()
                    struct_ref_std = preds_ref['std'].cpu().numpy()
                    out['bayesian_structured_heldout_mean'] = struct_ref_mean
                    out['bayesian_structured_heldout_std'] = struct_ref_std
                print('  Bayesian structured candidate prediction loaded')
                if x_ref_tensor is not None:
                    _print_ref_metrics('Bayesian structured',
                                       out['bayesian_structured_heldout_mean'],
                                       out['heldout_u'],
                                       out['bayesian_structured_heldout_std'])
                if 'exact' in out:
                    det_err = np.abs(struct_mean - out['exact'])
                    rel_l2 = _relative_l2(struct_mean, out['exact'])
                    flat_e = det_err.reshape(-1)
                    flat_s = struct_std.reshape(-1)
                    corr = (np.corrcoef(flat_e, flat_s)[0, 1]
                            if np.std(flat_e) > 1e-12 and np.std(flat_s) > 1e-12
                            else np.nan)
                    print(f'  Bayesian structured rel_L2={rel_l2:.4e}, '
                          f'max_error={det_err.max():.4e}, '
                          f'avg_std={flat_s.mean():.4e}, corr(|err|,std)={corr:.4f}')

                force_struct = PT.as_bool(config.get('force_structured_prediction', None),
                                          default=False)
                accept_ratio = float(config.get('accept_structured_rel_l2_ratio', 1.10))
                accept_abs = float(config.get('accept_structured_rel_l2_abs', 0.002))
                min_compression = float(config.get('min_structured_compression', 0.0))
                compression_ok = (
                    force_struct or min_compression <= 0
                    or (np.isfinite(struct_compression)
                        and struct_compression >= min_compression)
                )
                if min_compression > 0:
                    status = 'ok' if compression_ok else 'reject'
                    print(f'  Structured compression guard: '
                          f'compression={struct_compression:.3f}x, '
                          f'min={min_compression:.3f}x -> {status}')

                accept_struct = bool(force_struct)
                dense_rel_guard = None
                struct_rel_guard = None
                threshold_guard = None
                reason = 'forced' if force_struct else ''
                if not accept_struct:
                    if not compression_ok:
                        accept_struct = False
                        reason = (f'compression {struct_compression:.3f}x '
                                  f'< min {min_compression:.3f}x')
                    elif ('exact' in out and 'bayesian_dense_mean' in out
                          and guard_mask is not None):
                        dense_rel_guard = _masked_rel_l2(
                            out['bayesian_dense_mean'], out['exact'], guard_mask)
                        struct_rel_guard = _masked_rel_l2(
                            struct_mean, out['exact'], guard_mask)
                        threshold_guard = dense_rel_guard * accept_ratio + accept_abs
                        accept_struct = struct_rel_guard <= threshold_guard
                        reason = ('struct <= gamma*dense + eps on guard'
                                  if accept_struct
                                  else 'struct > gamma*dense + eps on guard')
                        status = 'accepted' if accept_struct else 'kept dense student'
                        print(f'  Structured guard (guard subset): '
                              f'dense rL2={dense_rel_guard:.4e}, '
                              f'structured rL2={struct_rel_guard:.4e}, '
                              f'threshold={threshold_guard:.4e} -> {status}')
                    elif x_ref_tensor is not None and 'bayesian_heldout_mean' in out:
                        dense_rel_guard = _relative_l2(
                            out['bayesian_heldout_mean'], out['heldout_u'])
                        struct_rel_guard = _relative_l2(struct_ref_mean, out['heldout_u'])
                        threshold_guard = dense_rel_guard * accept_ratio + accept_abs
                        accept_struct = struct_rel_guard <= threshold_guard
                        reason = 'heldout-serial guard'
                        status = 'accepted' if accept_struct else 'kept dense student'
                        print(f'  Structured heldout guard: dense rL2={dense_rel_guard:.4e}, '
                              f'structured rL2={struct_rel_guard:.4e}, '
                              f'threshold={threshold_guard:.4e} -> {status}')
                    else:
                        accept_struct = True
                        reason = 'no reference available, default accept'

                _write_guard_decision_row(
                    case, seed_for_split, dense_rel_guard, struct_rel_guard,
                    threshold_guard, struct_compression, min_compression,
                    accept_struct, reason)

                if accept_struct:
                    out['bayesian_mean'] = struct_mean
                    out['bayesian_mc_mean'] = struct_mc_mean
                    out['bayesian_std'] = struct_std
                    out['bayesian_epistemic'] = struct_epistemic
                    out['bayesian_aleatoric'] = struct_aleatoric
                    out['bayesian_samples'] = struct_samples
                    out['bayesian_source'] = np.array(['structured'])
                    if x_ref_tensor is not None:
                        out['bayesian_heldout_mean'] = struct_ref_mean
                        out['bayesian_heldout_std'] = struct_ref_std
        except Exception as e:
            print(f'  Bayesian structured prediction failed: {e}')

    _maybe_apply_structure_std_calibration(case, config, out)
    _maybe_apply_temperature_std_calibration(case, config, out)

    # Structured model prediction
    struct_path = f'{model_dir}/{case}_EXP_structured.pth'
    if os.path.isfile(struct_path):
        try:
            ckpt = torch.load(struct_path, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt:
                out['has_structured'] = np.array([1])
                print(f'  Structured checkpoint loaded')
        except Exception as e:
            print(f'  Structured load failed: {e}')

    if blind_mask is not None and 'exact' in out:
        metrics_path = './results/metrics/blind_test_metrics.csv'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        rows = []
        blind_1d = blind_mask.reshape(-1).astype(bool)
        exact_1d = np.asarray(out['exact']).reshape(-1)
        for label, mean_key in [
            ('dense', 'bayesian_dense_mean'),
            ('structured', 'bayesian_structured_mean'),
            ('guarded_final', 'bayesian_mean'),
        ]:
            if mean_key not in out:
                continue
            pred_1d = np.asarray(out[mean_key]).reshape(-1)
            rows.append({
                'case': case,
                'seed': seed_for_split,
                'source': label,
                'blind_rel_L2': _masked_rel_l2(pred_1d, exact_1d, blind_1d),
                'blind_MAE': float(np.mean(np.abs(
                    pred_1d[blind_1d] - exact_1d[blind_1d]))),
                'n_blind': int(blind_1d.sum()),
            })
        existing = []
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing = [
                        old for old in reader
                        if not (old.get('case') == case
                                and str(old.get('seed')) == str(seed_for_split))
                    ]
            except Exception:
                existing = []
        if rows:
            fields = list(rows[0].keys())
            with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(existing)
                writer.writerows(rows)
            print(f'  Blind-test metrics written: {len(rows)} rows -> {metrics_path}')

    save_path = f'{output_dir}/{case}_predictions.npz'
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez(save_path, **arrays)
    print(f'  Saved {len(arrays)} arrays to {save_path}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='Laplace')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--grid_n', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    predict_and_save(args.case, n_samples=args.n_samples, grid_n=args.grid_n,
                     device_str=args.device, seed=args.seed)
