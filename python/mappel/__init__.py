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
except ImportError:
    pass
try:
    from _Gauss1DsMAP import Gauss1DsMAP
except ImportError:
    pass

from . MappelBase import MappelBase



