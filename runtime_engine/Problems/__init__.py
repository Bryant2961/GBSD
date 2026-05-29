"""
Problems package — DEPRECATED.

The canonical entry point is now:
    import Module.Training as Training
    task = Training.model('Laplace', 'EXP', student_type='mc_dropout')
    task.train()

This shim exists for backward compatibility with code that calls:
    from Problems import get_runner
    runner = get_runner('Laplace')
    runner.run_problem(config)
"""
import warnings

# Keep sub-module imports alive for any external code referencing them
from . import laplace
from . import burgers_inverse
from . import poisson

REGISTRY = {
    'Laplace':     laplace,
    'Burgers_inv': burgers_inverse,
    'Poisson':     poisson,
}


def get_runner(problem_name: str):
    """DEPRECATED: Use Module.Training.model() instead."""
    warnings.warn(
        "Problems.get_runner() is deprecated. "
        "Use: Training.model(ques_name, ini_num, student_type=...).train() instead.",
        DeprecationWarning, stacklevel=2)
    if problem_name not in REGISTRY:
        raise ValueError(f"Unknown problem '{problem_name}'. "
                         f"Choose from: {list(REGISTRY.keys())}")
    return REGISTRY[problem_name]
