import unittest

import torch
from deeplay.applications.application import Application
from deeplay.components.mlp import MultiLayerPerceptron
from deeplay.external.optimizers.adam import Adam
from deeplay.module import DeeplayModule
from deeplay.schedulers import ConstantScheduler


class TestConstantScheduler(unittest.TestCase):

    def test_scheduler_build(self):
        scheduler = ConstantScheduler(1.0)
        scheduler.build()

        self.assertEqual(scheduler._step, 0)
        self.assertIsNone(scheduler._x)

    def test_scheduler_step(self):
        scheduler = ConstantScheduler(1.0)
        scheduler.build()

        steps = [-999999999, -1, 0, 999, 9999999999999]
        for step in steps:
            value = scheduler(step)
            self.assertEqual(value, 1.0)

    def test_scheduler_attached_to_module(self):

        class Module(DeeplayModule):
            def __init__(self):
                super().__init__()
                self.x = ConstantScheduler(1.0)

        module = Module()

        # Before build, x is the scheduler
        self.assertIsInstance(module.x, ConstantScheduler)

        module.build()

        # After build, x is the value of the scheduler
        self.assertEqual(module.x, 1.0)

    def test_scheduler_attached_configure(self):

        class Module(DeeplayModule):
            def __init__(self):
                super().__init__()
                self.x = ConstantScheduler(1.0)

        module = Module()

        # Before build, x is the scheduler
        self.assertIsInstance(module.x, ConstantScheduler)

        module.x.configure(value=2.0)

        module.build()

        # After build, x is the value of the scheduler
        self.assertEqual(module.x, 2.0)

    def test_scheduler_trainer(self):

        class Module(Application):
            def __init__(self):
                super().__init__(optimizer=Adam(lr=1.0), loss=torch.nn.MSELoss())
                self.x = ConstantScheduler(1.0)
                self.net = MultiLayerPerceptron(1, [1], 1)

            def forward(_self, x):
                self.assertEqual(_self.x, 1.0)
                return _self.net(x) * _self.x

        module = Module()

        module.build()

        x = torch.randn(10, 1)
        y = torch.randn(10, 1)

        module.fit((x, y), max_steps=10)
        module._has_built = False

        self.assertEqual(module.x.trainer, module.trainer)
        self.assertEqual(module.x._step, 9)
        self.assertEqual(module.x._x, 1.0)
