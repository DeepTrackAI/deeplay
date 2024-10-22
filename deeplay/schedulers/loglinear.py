from . import BaseScheduler
import numpy as np


class LogLinearScheduler(BaseScheduler):
    """Scheduler that returns log-linearly changing value from start_value to end_value.

    For steps beyond n_steps, returns end_value."""

    def __init__(self, start_value, end_value, n_steps, on_epoch=False):
        super().__init__(on_epoch)
        assert np.sign(start_value) == np.sign(
            end_value
        ), "Start and end values must have the same sign"
        assert start_value != 0, "Start value must be non-zero"
        assert end_value != 0, "End value must be non-zero"
        assert n_steps > 0, "Number of steps must be greater than 0"
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def __call__(self, step):
        if step >= self.n_steps:
            return self.end_value
        return self.start_value * (self.end_value / self.start_value) ** (
            step / self.n_steps
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start_value}, {self.end_value}, {self.n_steps})"

    def __str__(self):
        return repr(self)
