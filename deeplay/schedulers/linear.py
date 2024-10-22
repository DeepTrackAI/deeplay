from .scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    """Scheduler that returns linearly changing value from start_value to end_value.

    For steps beyond n_steps, returns end_value.
    For steps before 0, returns start_value.

    Parameters
    ----------
    start_value : float
        Initial value of the scheduler.
    end_value : float
        Final value of the scheduler.
    n_steps : int
        Number of steps to reach end_value.
    on_epoch : bool
        If True, the step is taken from the epoch counter of the trainer.
        Otherwise, the step is taken from the global step counter of the trainer.
    """

    def __init__(self, start_value, end_value, n_steps, on_epoch=False):
        super().__init__(on_epoch)
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def __call__(self, step):
        if step < 0:
            return self.start_value
        if step >= self.n_steps:
            return self.end_value
        return (
            self.start_value + (self.end_value - self.start_value) * step / self.n_steps
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start_value}, {self.end_value}, {self.n_steps})"

    def __str__(self):
        return repr(self)
