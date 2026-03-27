import unittest
from deeplay.schedulers import BaseScheduler


class TestBaseScheduler(unittest.TestCase):

    def test_scheduler_defaults(self):
        scheduler = BaseScheduler()
        scheduler.build()

        self.assertEqual(scheduler._step, 0)
        self.assertIsNone(scheduler._x)
