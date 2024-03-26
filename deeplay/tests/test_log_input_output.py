import unittest
import deeplay as dl
import torch
import torch.nn as nn
import itertools


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


class LogModuleWrapper(dl.DeeplayModule):
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        return self.model(x)


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

    def test_log_input_output_wrapped(self):

        use_selection = [False, True]
        before_wrap_do = ["build", "create", None]
        set_log_before_wrap = [False, True]
        after_wrap_do = ["build", "create"]

        for (
            use_selection,
            set_log_before_wrap,
            before_wrap_do,
            after_wrap_do,
        ) in itertools.product(
            use_selection, set_log_before_wrap, before_wrap_do, after_wrap_do
        ):
            with self.subTest(
                set_log_before_wrap=set_log_before_wrap,
                use_selection=use_selection,
                before_wrap_do=before_wrap_do,
                after_wrap_do=after_wrap_do,
            ):

                child = LogModule()

                if set_log_before_wrap or before_wrap_do is not None:
                    if use_selection:
                        child["cnn"].log_input("cnn_in")
                        child["cnn"].log_output("cnn_out")
                        child["layer"].log_input("layer_in")
                        child["layer"].log_output("layer_out")
                    else:
                        child.cnn.log_input("cnn_in")
                        child.cnn.log_output("cnn_out")
                        child.layer.log_input("layer_in")
                        child.layer.log_output("layer_out")

                if before_wrap_do == "build":
                    child.build()
                elif before_wrap_do == "create":
                    child = child.create()

                model = LogModuleWrapper(child)

                if not (set_log_before_wrap or before_wrap_do is not None):
                    if use_selection:
                        model["model", "cnn"].log_input("cnn_in")
                        model["model", "cnn"].log_output("cnn_out")
                        model["model", "layer"].log_input("layer_in")
                        model["model", "layer"].log_output("layer_out")
                    else:
                        model.model.cnn.log_input("cnn_in")
                        model.model.cnn.log_output("cnn_out")
                        model.model.layer.log_input("layer_in")
                        model.model.layer.log_output("layer_out")

                if after_wrap_do == "build":
                    model.build()
                elif after_wrap_do == "create":
                    model = model.create()

                model.forward(torch.rand(1, 1))

                self.assertEqual(len(model.logs), 4)
                self.assertEqual(
                    model.model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0]
                )
                self.assertEqual(model.model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
                self.assertEqual(
                    model.model.layer_in[0, 0], model.logs["layer_in"][0][0, 0]
                )
                self.assertEqual(
                    model.model.layer_out[0, 0], model.logs["layer_out"][0, 0]
                )

                model.forward(torch.rand(1, 1))

                self.assertEqual(len(model.logs), 4)
                self.assertEqual(
                    model.model.cnn_in[0, 0], model.logs["cnn_in"][0][0, 0]
                )
                self.assertEqual(model.model.cnn_out[0, 0], model.logs["cnn_out"][0, 0])
                self.assertEqual(
                    model.model.layer_in[0, 0], model.logs["layer_in"][0][0, 0]
                )
                self.assertEqual(
                    model.model.layer_out[0, 0], model.logs["layer_out"][0, 0]
                )
