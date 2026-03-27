from . import BaseScheduler


class ConstantScheduler(BaseScheduler):
    """Sheduler that returns constant value."""

    def __init__(self, value, on_epoch=False):
        super().__init__(on_epoch)
        self.value = value

    def __call__(self, step):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __str__(self):
        return repr(self)
