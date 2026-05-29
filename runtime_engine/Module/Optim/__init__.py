"""
Module.Optim — DEPRECATED / EXPERIMENTAL

This sub-package contains experimental optimized training techniques
(Fourier features, adaptive loss weighting, causal training, RAR, L-BFGS)
that are NOT part of the official GBSD reproduction path.

These modules are preserved for research use but are not called by the
main Training.model pipeline. They may be removed in a future version.
"""
import warnings
warnings.warn(
    "Module.Optim is experimental and not part of the official GBSD reproduction path. "
    "Use Module.Training.model() for the standard pipeline.",
    DeprecationWarning, stacklevel=2)

from . import OptimizedPINN
from . import OptimizedTraining
