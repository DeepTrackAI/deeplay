from .scheduler import BaseScheduler


class SchedulerSequence(BaseScheduler):
    """Scheduler that returns value from one of the schedulers in the chain.

    The scheduler is chosen based on the current step.
    """

    def __init__(self, on_epoch=False):
        super().__init__(on_epoch)
        self.schedulers = []

    def add(self, scheduler, n_steps=None):
        if n_steps is None:
            assert hasattr(
                scheduler, "n_steps"
            ), "For a scheduler without n_steps, n_steps must be specified"
            n_steps = scheduler.n_steps

        self.schedulers.append((n_steps, scheduler))

    def __call__(self, step):
        for n_steps, scheduler in self.schedulers:
            if step < n_steps:
                return scheduler(step)
            step -= n_steps

        final_step, final_scheduler = self.schedulers[-1]
        return final_scheduler(final_step + step)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schedulers})"

    def __str__(self):
        return repr(self)
