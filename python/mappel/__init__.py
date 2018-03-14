from . methods import _WrapModelClass

try:
    from _Gauss1DMLE import Gauss1DMLE
    _WrapModelClass(Gauss1DMLE)
except ImportError:
    pass

try:
    from _Gauss1DMAP import Gauss1DMAP
    _WrapModelClass(Gauss1DMAP)
except ImportError:
    pass

try:
    from _Gauss1DsMLE import Gauss1DsMLE
    _WrapModelClass(Gauss1DsMLE)
except ImportError:
    pass
try:
    from _Gauss1DsMAP import Gauss1DsMAP
    _WrapModelClass(Gauss1DsMAP)
except ImportError:
    pass

try:
    from _Gauss2DMLE import Gauss2DMLE
    _WrapModelClass(Gauss2DMLE)
except ImportError:
    pass

try:
    from _Gauss2DMAP import Gauss2DMAP
    _WrapModelClass(Gauss2DMAP)
except ImportError:
    pass

try:
    from _Gauss2DsMLE import Gauss2DsMLE
    _WrapModelClass(Gauss2DsMLE)
except ImportError:
    pass
try:
    from _Gauss2DsMAP import Gauss2DsMAP
    _WrapModelClass(Gauss2DsMAP)
except ImportError:
    pass
