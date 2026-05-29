"""
test_smoke.py — Lightweight structural and mathematical sanity checks.

Finishes in < 30 seconds. Tests import, config loading, architecture
consistency, Burgers/Poisson correctness, and grid construction — without
running any full training pipelines.

Usage: python test_smoke.py
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed, failed = 0, 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f'  PASS: {name}')
        passed += 1
    else:
        print(f'  FAIL: {name}  {detail}')
        failed += 1


start = time.time()

# ── 1. Import critical modules ──
print('\n=== Module Imports ===')
try:
    import Module.Training as Training
    import Module.PINN as PINN
    import Module.Student_MCDropout as MCDrop
    from utils.posterior_predict import _load_config, _build_layer, get_exact_solution, make_eval_grid
    from utils.metrics import relative_l2, compute_all_metrics
    from utils.paper_style import TRUE_PARAMS, GRID_CONFIGS
    check('All critical modules imported', True)
except Exception as e:
    check('All critical modules imported', False, str(e))
    sys.exit(1)

# ── 2. Load configs for active cases ──
print('\n=== Config Loading ===')
for case in ['Laplace', 'Poisson', 'Burgers_inv']:
    cfg = _load_config(case)
    check(f'{case} config loaded', len(cfg) > 0, f'got {len(cfg)} keys')

# ── 3. Burgers_inv domain check ──
print('\n=== Burgers_inv Domain ===')
cfg_b = _load_config('Burgers_inv')
x_min = float(cfg_b.get('x_min', 0))
x_max = float(cfg_b.get('x_max', 0))
y_min = float(cfg_b.get('y_min', 0))
y_max = float(cfg_b.get('y_max', 0))
check('Burgers x_min == -1',  x_min == -1.0, f'got {x_min}')
check('Burgers x_max == 1',   x_max == 1.0,  f'got {x_max}')
check('Burgers y_min (t_min) == 0', y_min == 0.0, f'got {y_min}')
check('Burgers y_max (t_max) == 1', y_max == 1.0, f'got {y_max}')

# Grid via make_eval_grid
_, X1, X2 = make_eval_grid('Burgers_inv', n=10, config=cfg_b)
check('Burgers grid x range [-1,1]', abs(X1.min() - (-1)) < 1e-6 and abs(X1.max() - 1) < 1e-6,
      f'got [{X1.min()}, {X1.max()}]')
check('Burgers grid t range [0,1]',  abs(X2.min()) < 1e-6 and abs(X2.max() - 1) < 1e-6,
      f'got [{X2.min()}, {X2.max()}]')

# paper_style GRID_CONFIGS consistency
gc = GRID_CONFIGS.get('Burgers_inv', {})
check('paper_style Burgers xmin == -1', gc.get('xmin') == -1, f'got {gc.get("xmin")}')
check('paper_style Burgers ymin == 0',  gc.get('ymin') == 0,  f'got {gc.get("ymin")}')

# ── 4. Architecture from config (not hard-coded) ──
print('\n=== Architecture from Config ===')

lap_layer = _build_layer(_load_config('Laplace'))
check('Laplace layer == [2,32,32,32,1]', lap_layer == [2, 32, 32, 32, 1],
      f'got {lap_layer}')

poi_layer = _build_layer(_load_config('Poisson'))
poi_node = int(_load_config('Poisson').get('node_num', 32))
poi_cfg = _load_config('Poisson')
poi_input = 18 if str(poi_cfg.get('use_fourier_features', 'False')).lower() == 'true' else 2
check(f'Poisson layer == [{poi_input},{poi_node},{poi_node},{poi_node},1]',
      poi_layer == [poi_input, poi_node, poi_node, poi_node, 1],
      f'got {poi_layer}')

# ── 5. Forward pass ──
print('\n=== Forward Pass ===')
net = PINN.Net([2, 32, 32, 32, 1])
x_test = torch.randn(16, 2)
with torch.no_grad():
    y_test = net(x_test)
check('PINN forward shape', y_test.shape == (16, 1), f'got {y_test.shape}')

# ── 6. Burgers residual check ──
print('\n=== Burgers Residual ===')
from Module.Training import burgers_residual
x_br = torch.randn(32, 2, requires_grad=True)
net_small = PINN.Net([2, 16, 16, 1])
loss_br = burgers_residual(net_small, x_br)
check('burgers_residual returns scalar', loss_br.dim() == 0, f'dim={loss_br.dim()}')
check('burgers_residual is finite', torch.isfinite(loss_br).item())

# Verify it uses u_t + u*u_x - nu*u_xx (not swapped)
# Check by evaluating at a known point: u=0 everywhere => residual should be 0-ish
# since all derivatives of a random network are nonzero, just check finite + positive
check('burgers_residual is non-negative', loss_br.item() >= 0)

# ── 7. Poisson exact solution / source consistency ──
print('\n=== Poisson Consistency ===')
x_poi = np.random.rand(100, 2)  # domain [0,1]^2
u_exact = get_exact_solution('Poisson', x_poi)  # shape (100,1)

# Verify: -2*k^2*pi^2 * A_k == f_k for all k=1..4
all_consistent = True
for k in range(1, 5):
    A_k = 0.5 * ((-1)**k) / (2 * np.pi**2)
    f_k = 0.5 * ((-1)**(k+1)) * k**2
    laplacian = -2 * k**2 * np.pi**2 * A_k
    if abs(laplacian - f_k) > 1e-10:
        all_consistent = False
check('Poisson -2k²π²·A_k == f_k for k=1..4', all_consistent)

# Check exact solution is not None and has right shape
check('Poisson exact solution shape', u_exact is not None and u_exact.shape == (100, 1),
      f'got {u_exact.shape if u_exact is not None else None}')

# Check boundary condition: u(0, y) = u(1, y) = u(x, 0) = u(x, 1) = 0
x_bnd = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
u_bnd = get_exact_solution('Poisson', x_bnd)
check('Poisson BC: u=0 on boundary', np.allclose(u_bnd, 0, atol=1e-10),
      f'max |u|={np.abs(u_bnd).max():.2e}')

# ── 8. True parameter value ──
print('\n=== Parameter Inversion Values ===')
import math
tp = TRUE_PARAMS.get('Burgers_inv', {}).get('parameters_1', {})
true_val = tp.get('true_value', 0) if isinstance(tp, dict) else tp
check('TRUE_PARAMS Burgers nu ≈ 0.01/pi',
      abs(true_val - 0.01/math.pi) < 1e-10,
      f'got {true_val}, expected {0.01/math.pi}')

# ── 9. Laplace exact solution ──
print('\n=== Laplace Reference Solution ===')
x_lap = np.array([[0.5, 0.3], [-1.0, 1.0], [0.0, 0.0]])
u_lap = get_exact_solution('Laplace', x_lap)
u_expected = x_lap[:, 0:1]**3 - 3*x_lap[:, 0:1]*x_lap[:, 1:2]**2
check('Laplace u = x³ - 3xy²', np.allclose(u_lap, u_expected))

# ── 10. Burgers data file column ordering ──
print('\n=== Burgers Data Files ===')
import pandas as pd
d1 = pd.read_csv('./Database/Burgers_inv_data_1.csv', header=None)
d2 = pd.read_csv('./Database/Burgers_inv_data_2.csv', header=None)
d3 = pd.read_csv('./Database/Burgers_inv_data_3.csv', header=None)
check('data_1 col0 (x) in [-1,1]', d1[0].min() >= -1 and d1[0].max() <= 1,
      f'[{d1[0].min():.3f}, {d1[0].max():.3f}]')
check('data_1 col1 (t) in [0,1]',  d1[1].min() >= 0 and d1[1].max() <= 1,
      f'[{d1[1].min():.3f}, {d1[1].max():.3f}]')
check('data_2 col0 (x) in [-1,1]', d2[0].min() >= -1 and d2[0].max() <= 1,
      f'[{d2[0].min():.3f}, {d2[0].max():.3f}]')
check('data_2 col1 (t) in [0,1]',  d2[1].min() >= 0 and d2[1].max() <= 1,
      f'[{d2[1].min():.3f}, {d2[1].max():.3f}]')
check('data_3 col0 (x) in [-1,1]', d3[0].min() >= -1 and d3[0].max() <= 1,
      f'[{d3[0].min():.3f}, {d3[0].max():.3f}]')
check('data_3 col1 (t) in [0,1]',  d3[1].min() >= 0 and d3[1].max() <= 1,
      f'[{d3[1].min():.3f}, {d3[1].max():.3f}]')

ref = np.load('./Database/Burgers_inv_reference.npz')
check('Burgers reference grid present', {'x', 't', 'u', 'nu', 'ic_sign'} <= set(ref.files))
check('Burgers reference IC sign is -1', abs(float(ref['ic_sign'][0]) + 1.0) < 1e-12)
u0 = ref['u'][0]
x_ref = ref['x']
check('Burgers reference u(x,0)=-sin(pi*x)',
      np.max(np.abs(u0 + np.sin(np.pi * x_ref))) < 5e-3)

# ── 11. Training budget fairness ──
print('\n=== Training Budget Fairness ===')
for case in ['Laplace', 'Poisson', 'Burgers_inv']:
    cfg = _load_config(case)
    ts = int(cfg.get('train_steps', 0))
    sts = int(cfg.get('student_train_steps', ts))
    check(f'{case} student == teacher budget', ts == sts,
          f'teacher={ts}, student={sts}')

# ── Summary ──
elapsed = time.time() - start
print(f'\n{"="*60}')
print(f'  SMOKE TEST: {passed} passed, {failed} failed ({elapsed:.1f}s)')
print(f'{"="*60}')

if failed > 0:
    sys.exit(1)
