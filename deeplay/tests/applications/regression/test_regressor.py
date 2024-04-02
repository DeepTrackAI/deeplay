import io
import unittest
import torch
import torch.nn as nn


from deeplay.applications.regression import Regressor
from deeplay.components import MultiLayerPerceptron
from deeplay.trainer import Trainer


class TestRegressor(unittest.TestCase):
    def test_regressor(self):

        creation_modes = ["build", "create"]

        for creation_mode in creation_modes:
            with self.subTest(creation_mode=creation_mode):
                net = MultiLayerPerceptron(1, [], 1)
                model = Regressor(net)
                model = getattr(model, creation_mode)()

                # forward pass
                x = torch.rand(1, 1)
                y = model(x)
                self.assertEqual(y.shape, (1, 1))

                # test is correct loss function
                self.assertIsInstance(model.loss, nn.L1Loss)

                # test is correct optimizer
                optimizer = model.configure_optimizers()
                self.assertIsInstance(optimizer, torch.optim.Adam)

                # test that optimizer updates the model

                layer_params = [
                    p.clone() for p in model.model.blocks[0].layer.parameters()
                ]
                optimizer.zero_grad()
                (y - 1).backward()
                optimizer.step()
                new_layer_params = model.model.blocks[0].layer.parameters()
                for param, new_param in zip(layer_params, new_layer_params):
                    self.assertTrue(torch.abs(param - new_param).sum() > 0)

    def test_regressor_train_preprocess(self):
        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).build()

        x = torch.rand(1, 1)
        y = torch.zeros(1, 1)
        x1, y1 = model.train_preprocess((x, y))

        self.assertTrue(torch.allclose(x, x1))
        self.assertTrue(torch.allclose(y, y1))

    def test_regressor_val_preprocess(self):
        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).build()

        x = torch.rand(1, 1)
        y = torch.zeros(1, 1)
        x1, y1 = model.val_preprocess((x, y))

        self.assertTrue(torch.allclose(x, x1))
        self.assertTrue(torch.allclose(y, y1))

    def test_regressor_test_preprocess(self):
        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).build()

        x = torch.rand(1, 1)
        y = torch.zeros(1, 1)
        x1, y1 = model.test_preprocess((x, y))

        self.assertTrue(torch.allclose(x, x1))
        self.assertTrue(torch.allclose(y, y1))

    def test_regressor_compute_loss(self):
        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).build()

        y_hat = torch.ones(1, 1)
        y = torch.zeros(1, 1)
        loss = model.compute_loss(y_hat, y)
        self.assertEqual(loss, 1.0)

    def test_regressor_fit_small(self):
        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).build()

        xtrain = torch.rand(1, 1)
        ytrain = torch.zeros(1, 1)

        xval = torch.rand(1, 1)
        yval = torch.zeros(1, 1)

        h = model.fit(
            train_data=(xtrain, ytrain),
            val_data=(xval, yval),
            max_epochs=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        self.assertIn("train_loss_epoch", h.history)
        self.assertIn("val_loss_epoch", h.history)
        self.assertIn("train_loss_step", h.step_history)

    def test_regressor_save_and_load(self):

        x_data = torch.rand(1, 1)

        net = MultiLayerPerceptron(1, [], 1)
        model = Regressor(net).create()

        y = model(x_data)

        file = io.BytesIO()
        torch.save(model.state_dict(), file)

        model = Regressor(net).create()
        file.seek(0)
        model.load_state_dict(torch.load(file))

        y1 = model(x_data)

        self.assertTrue(torch.allclose(y, y1))
