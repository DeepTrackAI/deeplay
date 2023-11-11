import warnings

# temporary imports until we implement our own

from lightning import Trainer
from torch.utils.data import DataLoader

# Filter out warnings from lazy torch modules
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

# from . import core, schedulers
# from .core import Config, Ref, OutputOf, Layer, MultiInputLayer

# from .components import *
# from .applications import *

from .meta import ExtendedConstructorMeta
from .module import DeeplayModule
from .list import LayerList, Sequential
from .external import *
from .blocks import *
from .components import *


from . import tests
