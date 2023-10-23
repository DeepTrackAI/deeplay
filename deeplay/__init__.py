import warnings

# Filter out warnings from lazy torch modules
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

from . import core
from .core import Config, Ref, OutputOf, Layer, MultiInputLayer

from .components import *
from .applications import *

from . import datasets, tests

# temporary imports until we implement our own
from lightning import Trainer
from torch.utils.data import DataLoader
