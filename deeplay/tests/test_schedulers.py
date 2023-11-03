import unittest
from ..schedulers import (
    LinearScheduler,
    ConstantScheduler,
    LogLinearScheduler,
    SchedulerSequence,
)
import numpy as np


class TestConstentScheduler(unittest.TestCase):
    def test_constant_scheduler(self):
        constant_scheduler = ConstantScheduler(1)
        self.assertEqual(constant_scheduler(0), 1)
        self.assertEqual(constant_scheduler(1), 1)
        self.assertEqual(constant_scheduler(2), 1)
        self.assertEqual(constant_scheduler(1000), 1)


class TestLinearScheduler(unittest.TestCase):
    def test_linear_scheduler(self):
        linear_scheduler = LinearScheduler(1, 2, 10)
        self.assertEqual(linear_scheduler(0), 1)
        self.assertEqual(linear_scheduler(1), 1.1)
        self.assertEqual(linear_scheduler(2), 1.2)
        self.assertEqual(linear_scheduler(3), 1.3)
        self.assertEqual(linear_scheduler(4), 1.4)
        self.assertEqual(linear_scheduler(5), 1.5)
        self.assertEqual(linear_scheduler(6), 1.6)
        self.assertEqual(linear_scheduler(7), 1.7)
        self.assertEqual(linear_scheduler(8), 1.8)
        self.assertEqual(linear_scheduler(9), 1.9)
        self.assertEqual(linear_scheduler(10), 2)
        self.assertEqual(linear_scheduler(11), 2)
        self.assertEqual(linear_scheduler(1000), 2)

    def test_linear_scheduler_no_steps(self):
        linear_scheduler = LinearScheduler(1, 2, 0)
        self.assertEqual(linear_scheduler(0), 2)
        self.assertEqual(linear_scheduler(1), 2)
        self.assertEqual(linear_scheduler(2), 2)


class TestLogLinearScheduler(unittest.TestCase):
    def test_loglinear_scheduler(self):
        loglinear = np.logspace(-1, 1, 11, endpoint=True)
        loglinear_scheduler = LogLinearScheduler(0.1, 10, 10)

        for i in range(10):
            self.assertAlmostEqual(loglinear_scheduler(i), loglinear[i], msg=f"i={i}")
        self.assertEqual(loglinear_scheduler(11), 10)
        self.assertEqual(loglinear_scheduler(1000), 10)

    def test_loglinear_scheduler_no_steps(self):
        loglinear_scheduler = LogLinearScheduler(1, 2, 0)
        self.assertEqual(loglinear_scheduler(0), 2)
        self.assertEqual(loglinear_scheduler(1), 2)
        self.assertEqual(loglinear_scheduler(2), 2)


class TestSchedulerSequence(unittest.TestCase):
    def test_scheduler_sequence(self):
        scheduler_sequence = SchedulerSequence()
        scheduler_sequence.add(ConstantScheduler(1), 10)
        scheduler_sequence.add(LinearScheduler(1, 2, 10), 10)

        for i in range(10):
            self.assertEqual(scheduler_sequence(i), 1)

        for i in range(10, 20):
            self.assertEqual(scheduler_sequence(i), 1 + (i - 10) / 10)

        self.assertEqual(scheduler_sequence(20), 2)
        self.assertEqual(scheduler_sequence(21), 2)
