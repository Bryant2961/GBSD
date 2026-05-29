# Compatibility shim — canonical location is now Module.UncertaintyEstimation
import warnings as _w
_w.warn("Module.UQ is deprecated; import Module.UncertaintyEstimation directly.",
        DeprecationWarning, stacklevel=2)
from Module import UncertaintyEstimation
