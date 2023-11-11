from .scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    """Scheduler that returns linearly changing value from start_value to end_value.

    For steps beyond n_steps, returns end_value."""

    def __init__(self, start_value, end_value, n_steps, trainer=None, on_epoch=False):
        super().__init__(on_epoch)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def __call__(self, step):
        if step >= self.n_steps:
            return self.end_value
        return (
            self.start_value + (self.end_value - self.start_value) * step / self.n_steps
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start_value}, {self.end_value}, {self.n_steps})"

    def __str__(self):
        return repr(self)
