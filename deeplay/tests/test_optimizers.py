import deeplay as dl
import torch
import torch.nn as nn

import unittest


class TestOptimizers(unittest.TestCase):
    def test_optimizer_can_init(self):
        optimizer = dl.Optimizer(torch.optim.Adam, lr=0.1)

    def test_adam_can_init(self):
        optimizer = dl.Adam(lr=0.1)

    def test_sgd_can_init(self):
        optimizer = dl.SGD(lr=0.1)

    def test_rmsprop_can_init(self):
        optimizer = dl.RMSprop()

    def test_optimizer_can_build(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Optimizer(torch.optim.Adam, lr=0.1, params=net.parameters())
        optimizer = optimizer.build()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_adam_can_build(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Adam(lr=0.1, params=net.parameters())
        optimizer = optimizer.build()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_sgd_can_build(self):
        net = nn.Linear(10, 20)
        optimizer = dl.SGD(lr=0.1, params=net.parameters())
        optimizer = optimizer.build()

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_rmsprop_can_build(self):
        net = nn.Linear(10, 20)
        optimizer = dl.RMSprop(lr=0.1, params=net.parameters())
        optimizer = optimizer.build()

        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_optimizer_can_create(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Optimizer(torch.optim.Adam, lr=0.1, params=net.parameters())
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_adam_can_create(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Adam(lr=0.1, params=net.parameters())
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_sgd_can_create(self):
        net = nn.Linear(10, 20)
        optimizer = dl.SGD(lr=0.1, params=net.parameters())
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_rmsprop_can_create(self):
        net = nn.Linear(10, 20)
        optimizer = dl.RMSprop(lr=0.1, params=net.parameters())
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_optimizer_can_configure(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Optimizer(torch.optim.Adam, lr=0.1, params=net.parameters())
        optimizer.configure(lr=0.2)
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.2)

    def test_adam_can_configure(self):
        net = nn.Linear(10, 20)
        optimizer = dl.Adam(lr=0.1, params=net.parameters())
        optimizer.configure(lr=0.2)
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults["lr"], 0.2)

    def test_sgd_can_configure(self):
        net = nn.Linear(10, 20)
        optimizer = dl.SGD(lr=0.1, params=net.parameters())
        optimizer.configure(lr=0.2)
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.2)

    def test_rmsprop_can_configure(self):
        net = nn.Linear(10, 20)
        optimizer = dl.RMSprop(lr=0.1, params=net.parameters())
        optimizer.configure(lr=0.2)
        optimizer = optimizer.create()

        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["lr"], 0.2)

    def test_optimizer_build_in_application(self):
        net = nn.Linear(10, 20)
        application = dl.Classifier(net, optimizer=dl.Adam(lr=0.1))
        application = application.build()

        self.assertIsInstance(application.optimizer, torch.optim.Adam)
        self.assertEqual(application.optimizer.defaults["lr"], 0.1)
        self.assertListEqual(
            list(application.optimizer.param_groups[0]["params"]),
            list(net.parameters()),
        )

    def test_optimizer_create_in_application(self):
        net = nn.Linear(10, 20)
        application = dl.Classifier(net, optimizer=dl.Adam(lr=0.1))
        application = application.create()

        self.assertIsInstance(application.optimizer, torch.optim.Adam)
        self.assertEqual(application.optimizer.defaults["lr"], 0.1)
        self.assertListEqual(
            list(application.optimizer.param_groups[0]["params"]),
            list(net.parameters()),
        )

    def test_optimizer_configure_in_application(self):
        net = nn.Linear(10, 20)
        application = dl.Classifier(net, optimizer=dl.Adam(lr=0.1))
        application.optimizer.configure(lr=0.2)
        application = application.create()

        self.assertIsInstance(application.optimizer, torch.optim.Adam)
        self.assertEqual(application.optimizer.defaults["lr"], 0.2)
        self.assertListEqual(
            list(application.optimizer.param_groups[0]["params"]),
            list(net.parameters()),
        )
