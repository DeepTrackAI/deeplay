import lightning as L

from deeplay.module import DeeplayModule
from deeplay.trainer import Trainer


class BaseScheduler(DeeplayModule, L.LightningModule):
    """Base class for annealers."""

    step: int

    def __init__(self, on_epoch=False):
        super().__init__()
        self.on_epoch = on_epoch
        self._step = 0
        self._x = None

    def set_step(self, step):
        self._step = step
        self._x = self(step)

    def update(self):
        current_step = self._step

        if self._trainer:
            updated_step = (
                self.trainer.current_epoch
                if self.on_epoch
                else self.trainer.global_step
            )
        else:
            updated_step = self._step

        if updated_step != current_step or self._x is None:
            self.set_step(updated_step)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self.update()
        return self._x

    def __set__(self, obj, value):
        self._x = value

    def __call__(self, step):
        raise NotImplementedError
