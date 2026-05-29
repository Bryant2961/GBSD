# Compatibility shim — canonical locations are now Module.Student_MCDropout / Module.Student_VIBNN
import warnings as _w
_w.warn("Module.Bayesian is deprecated; import Module.Student_MCDropout / Module.Student_VIBNN directly.",
        DeprecationWarning, stacklevel=2)
from Module import Student_MCDropout
from Module import Student_VIBNN
