# Compatibility shim — canonical location is now Module.BayesianVis
import warnings as _w
_w.warn("Module.Vis is deprecated; import Module.BayesianVis directly.",
        DeprecationWarning, stacklevel=2)
from Module import BayesianVis as Visualization
