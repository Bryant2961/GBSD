# utils/blind_split.py — Fixed-seed guard-validation / blind-test split
"""
Pre-declared, fixed-seed evaluation-grid split shared by:
  - the in-pipeline guard decision (utils/posterior_predict.py)
  - the std calibration fit/eval (utils/posterior_predict.py)
  - the audit and summary scripts (Supplementary experiment/...)

Design choices
--------------
* The split is generated ONCE per (case, seed) and saved to:
      ./results/splits/<case>_seed<seed>.npz
  containing `guard_mask`, `blind_test_mask`, `seed`, `case`, `n`,
  `guard_fraction`, `created_at`.
* Once the file exists, it is loaded; it is never regenerated. The mask
  therefore travels with the run and gets archived by archive_run_outputs.
* The split is on flattened grid indices, so it works for any
  rectangular evaluation grid: Laplace/Poisson 80x80, Burgers 80x80, etc.
* For Burgers we already have a separate held-out CSV serial (used by
  std calibration via std_calibration_use_heldout). The blind_mask here
  is in addition to that and is used for the dense-grid evaluation.

API
---
    mask = ensure_split(case, seed, n_total)            # creates if missing
    guard_mask = mask['guard_mask']
    blind_test_mask = mask['blind_test_mask']

The reviewer requirement is: guard_mask is the only thing the guard
decision and std calibration may read from the exact array.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np


SPLIT_DIR = './results/splits'
DEFAULT_GUARD_FRACTION = 0.30
# Case-specific offset so cases do not share the same permutation.
_CASE_OFFSET = {'Laplace': 101, 'Poisson': 211, 'Burgers_inv': 307}


def split_path(case: str, seed: int) -> str:
    return os.path.join(SPLIT_DIR, f'{case}_seed{int(seed)}.npz')


def _build_masks(n_total: int, seed: int, case: str,
                 guard_fraction: float) -> Dict[str, np.ndarray]:
    if guard_fraction <= 0.0 or guard_fraction >= 1.0:
        raise ValueError('guard_fraction must be in (0, 1)')
    rng = np.random.default_rng(_CASE_OFFSET.get(case, 401) + 1009 * int(seed))
    perm = rng.permutation(int(n_total))
    n_guard = max(10, int(round(n_total * guard_fraction)))
    if n_total - n_guard < 10:
        raise ValueError(f'n_total={n_total} too small for guard_fraction={guard_fraction}')
    guard_idx = perm[:n_guard]
    blind_idx = perm[n_guard:]
    guard_mask = np.zeros(n_total, dtype=bool)
    blind_mask = np.zeros(n_total, dtype=bool)
    guard_mask[guard_idx] = True
    blind_mask[blind_idx] = True
    return {'guard_mask': guard_mask, 'blind_test_mask': blind_mask}


def ensure_split(case: str, seed: int, n_total: int,
                 guard_fraction: float = DEFAULT_GUARD_FRACTION) -> Dict[str, np.ndarray]:
    """Return (and create if missing) the (guard, blind) split file."""
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = split_path(case, seed)
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as f:
            saved_n = (int(np.asarray(f['n']).reshape(-1)[0])
                       if 'n' in f.files else None)
            if saved_n is not None and saved_n != int(n_total):
                # Grid size changed (different preset). Regenerate.
                pass
            else:
                return {
                    'guard_mask': f['guard_mask'].astype(bool),
                    'blind_test_mask': f['blind_test_mask'].astype(bool),
                    'seed': int(np.asarray(f['seed']).reshape(-1)[0]),
                    'case': str(np.asarray(f['case']).reshape(-1)[0]),
                    'n': int(np.asarray(f['n']).reshape(-1)[0]),
                    'guard_fraction': float(
                        np.asarray(f['guard_fraction']).reshape(-1)[0]),
                    'created_at': str(
                        np.asarray(f['created_at']).reshape(-1)[0]),
                    'source': 'loaded',
                }
    masks = _build_masks(n_total, seed, case, guard_fraction)
    payload = {
        **masks,
        'seed': np.array([int(seed)]),
        'case': np.array(case),
        'n': np.array([int(n_total)]),
        'guard_fraction': np.array([float(guard_fraction)]),
        'created_at': np.array(time.strftime('%Y-%m-%dT%H:%M:%S')),
    }
    np.savez(path, **payload)
    return {
        **masks,
        'seed': int(seed),
        'case': case,
        'n': int(n_total),
        'guard_fraction': float(guard_fraction),
        'created_at': payload['created_at'].item(),
        'source': 'created',
    }


def masked_rel_l2(pred: np.ndarray, exact: np.ndarray, mask: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float).reshape(-1)[mask]
    e = np.asarray(exact, dtype=float).reshape(-1)[mask]
    if p.size == 0 or e.size == 0:
        return float('nan')
    denom = float(np.sum(e ** 2))
    if denom <= 0.0:
        return float('nan')
    return float(np.sqrt(np.sum((p - e) ** 2) / max(denom, 1e-15)))
