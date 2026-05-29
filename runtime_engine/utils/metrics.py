# utils/metrics.py — Unified evaluation metrics
import numpy as np
import pandas as pd
from typing import Dict, Optional
import os

def relative_l2(pred, exact):
    return float(np.sqrt(np.sum((pred - exact)**2) / (np.sum(exact**2) + 1e-15)))

def mae(pred, exact):
    return float(np.mean(np.abs(pred - exact)))

def rmse(pred, exact):
    return float(np.sqrt(np.mean((pred - exact)**2)))

def mse_metric(pred, exact):
    return float(np.mean((pred - exact)**2))

def nll_gaussian(mean, std, exact):
    var = std**2 + 1e-8
    return float(np.mean(0.5 * np.log(2*np.pi*var) + (exact - mean)**2 / (2*var)))

def picp(mean, std, exact, confidence=0.95):
    z = {0.90:1.645, 0.95:1.96, 0.99:2.576}.get(confidence, 1.96)
    lower, upper = mean - z*std, mean + z*std
    return float(np.mean((exact >= lower) & (exact <= upper)))

def mpiw_metric(std, confidence=0.95):
    z = {0.90:1.645, 0.95:1.96, 0.99:2.576}.get(confidence, 1.96)
    return float(np.mean(2*z*std))

def calibration_error(mean, std, exact, n_bins=10):
    abs_err = np.abs(mean - exact).flatten()
    stds = std.flatten()
    bins = np.percentile(stds, np.linspace(0, 100, n_bins+1))
    ece = 0.0
    total = len(stds)
    for i in range(n_bins):
        mask = (stds >= bins[i]) & (stds < bins[i+1] + 1e-10)
        if mask.sum() == 0: continue
        frac = mask.sum() / total
        ece += frac * abs(np.mean(abs_err[mask]) - np.mean(stds[mask]))
    return float(ece)

def compute_all_metrics(mean, std, exact, case='', method='', seed=0, parameter_error=None,
                        source='pre_reconstruction', run_id='', preset=''):
    row = {'case': case, 'method': method, 'seed': seed,
           'source': source, 'run_id': run_id, 'preset': preset}
    if exact is not None:
        flat_m, flat_e = mean.flatten(), exact.flatten()
        row['l2_error'] = relative_l2(flat_m, flat_e)
        row['mae'] = mae(flat_m, flat_e)
        row['rmse'] = rmse(flat_m, flat_e)
        row['mse'] = mse_metric(flat_m, flat_e)
        # Solution amplitude context
        u_max = float(max(abs(flat_e.max()), abs(flat_e.min())))
        row['u_max'] = u_max
        row['mae_over_umax'] = float(row['mae'] / u_max) if u_max > 0 else np.nan
    else:
        row['l2_error'] = row['mae'] = row['rmse'] = row['mse'] = np.nan
        row['u_max'] = row['mae_over_umax'] = np.nan
    if std is not None and exact is not None:
        flat_s = std.flatten()
        row['nll'] = nll_gaussian(flat_m, flat_s, flat_e)
        row['coverage_95'] = picp(flat_m, flat_s, flat_e, 0.95)
        row['avg_interval_width_95'] = mpiw_metric(flat_s, 0.95)
        row['calibration_error'] = calibration_error(flat_m, flat_s, flat_e)
        # Error-uncertainty correlation
        abs_err = np.abs(flat_m - flat_e)
        if flat_s.std() > 1e-12 and abs_err.std() > 1e-12:
            row['error_uncertainty_corr'] = float(np.corrcoef(abs_err, flat_s)[0, 1])
        else:
            row['error_uncertainty_corr'] = np.nan
        # Std statistics
        row['std_min'] = float(flat_s.min())
        row['std_max'] = float(flat_s.max())
        row['std_mean'] = float(flat_s.mean())
    elif std is not None:
        flat_s = std.flatten()
        row['std_min'] = float(flat_s.min())
        row['std_max'] = float(flat_s.max())
        row['std_mean'] = float(flat_s.mean())
        row['nll'] = row['coverage_95'] = row['avg_interval_width_95'] = np.nan
        row['calibration_error'] = row['error_uncertainty_corr'] = np.nan
    else:
        row['nll'] = row['coverage_95'] = row['avg_interval_width_95'] = np.nan
        row['calibration_error'] = row['error_uncertainty_corr'] = np.nan
        row['std_min'] = row['std_max'] = row['std_mean'] = np.nan
    row['parameter_error'] = parameter_error if parameter_error is not None else np.nan
    return row

def save_metrics(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Metrics saved to {path}')
