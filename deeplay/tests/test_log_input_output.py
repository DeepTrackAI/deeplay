import unittest
import deeplay as dl
import torch
import torch.nn as nn


class LogModule(dl.DeeplayModule):
    def __init__(self):
        self.cnn = dl.MultiLayerPerceptron(1, [1, 1], 1)
        self.layer = dl.Layer(nn.Linear, 1, 1)

        self.cnn_in = None
        self.cnn_out = None
        self.layer_in = None
        self.layer_out = None

    def forward(self, x):
        self.cnn_in = x
        x = self.cnn(x)
        self.cnn_out = x
        self.layer_in = x
        x = self.layer(x)
        self.layer_out = x
        return x


class TestLogInputOutput(unittest.TestCase):
    def test_log_input_output(self):
        model = LogModule()
        model.cnn.log_input("cnn_in")
        model.cnn.log_output("cnn_out")
        model.layer.log_input("layer_in")
        model.layer.log_output("layer_out")

        model.build()

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

    def test_log_input_output_create(self):
        model = LogModule()

        model.cnn.log_input("cnn_in")
        model.cnn.log_output("cnn_out")
        model.layer.log_input("layer_in")
        model.layer.log_output("layer_out")

        model = model.create()

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

    def test_log_input_output_selection(self):
        model = LogModule()

        model["cnn"].log_input("cnn_in")
        model["cnn"].log_output("cnn_out")
        model["layer"].log_input("layer_in")
        model["layer"].log_output("layer_out")

        model.build()

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

    def test_log_input_output_selection_create(self):
        model = LogModule()

        model["cnn"].log_input("cnn_in")
        model["cnn"].log_output("cnn_out")
        model["layer"].log_input("layer_in")
        model["layer"].log_output("layer_out")

        model = model.create()

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])

        model.forward(torch.rand(1, 1))

        self.assertEqual(len(model.logs), 4)
        self.assertEqual(model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0])
        self.assertEqual(model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
        self.assertEqual(model.layer_in[0, 0], model.logs["layer_in"][0][0, 0])
        self.assertEqual(model.layer_out[0, 0], model.logs["layer_out"][0, 0])
