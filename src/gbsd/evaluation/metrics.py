"""Shared metric definitions for official GBSD summaries."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def relative_l2(pred, target) -> float:
    pred_arr = as_array(pred)
    target_arr = as_array(target)
    denom = float(np.sum(target_arr**2))
    return float(np.sqrt(np.sum((pred_arr - target_arr) ** 2) / max(denom, 1e-15)))


def coverage95(pred, target, std) -> float:
    pred_arr = as_array(pred)
    target_arr = as_array(target)
    std_arr = np.maximum(as_array(std), 1e-12)
    covered = np.abs(pred_arr - target_arr) <= 1.96 * std_arr
    return float(np.mean(covered))


def average_interval_width(std) -> float:
    std_arr = np.maximum(as_array(std), 1e-12)
    return float(np.mean(2.0 * 1.96 * std_arr))


def error_std_corr(pred, target, std) -> float:
    err = np.abs(as_array(pred) - as_array(target))
    std_arr = as_array(std)
    if err.size < 2 or np.std(err) == 0.0 or np.std(std_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(err, std_arr)[0, 1])


def gaussian_nll(pred, target, std) -> float:
    pred_arr = as_array(pred)
    target_arr = as_array(target)
    std_arr = np.maximum(as_array(std), 1e-12)
    var = std_arr**2
    nll = 0.5 * (np.log(2.0 * math.pi * var) + ((target_arr - pred_arr) ** 2) / var)
    return float(np.mean(nll))

