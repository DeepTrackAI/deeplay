import unittest
import torch
import torch.nn as nn
from deeplay import (
    LayerList,
    DeeplayModule,
    Layer,
    LayerActivation,
    Sequential,
    Parallel,
)
import itertools

from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.list import ReferringLayerList


class Wrapper1(DeeplayModule):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(Layer(nn.Linear, i + 1, i + 2))

        self.layers = LayerList(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Wrapper2(DeeplayModule):
    def __init__(self, n_layers):
        super().__init__()
        layers = LayerList()
        for i in range(n_layers):
            layers.append(Layer(nn.Linear, i + 1, i + 2))

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Wrapper3(DeeplayModule):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = LayerList()
        for i in range(n_layers):
            self.layers.append(Layer(nn.Linear, i + 1, i + 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestLayerList(unittest.TestCase):
    def test_create_list(self):
        for Wrapper in [Wrapper1, Wrapper2, Wrapper3]:
            module = Wrapper(5)
            self.assertEqual(len(module.layers), 5, Wrapper)
            module.build()
            self.assertEqual(len(module.layers), 5, Wrapper)
            for i in range(5):
                self.assertEqual(module.layers[i].in_features, i + 1, Wrapper)
                self.assertEqual(module.layers[i].out_features, i + 2, Wrapper)

            x = torch.randn(10, 1)
            y = module(x)
            self.assertEqual(y.shape, (10, 6), Wrapper)

    def test_configure_list(self):
        for Wrapper in [Wrapper1, Wrapper2, Wrapper3]:
            module = Wrapper(5)
            module.layers[0].configure(in_features=2)
            module.build()
            self.assertEqual(len(module.layers), 5, Wrapper)
            self.assertEqual(module.layers[0].in_features, 2, Wrapper)

    def test_index_slice(self):
        for Wrapper in [Wrapper1, Wrapper2, Wrapper3]:
            module = Wrapper(5)
            module.layers[1:3].configure(in_features=2)
            module.build()
            self.assertEqual(len(module.layers), 5, Wrapper)
            self.assertEqual(module.layers[1].in_features, 2, Wrapper)
            self.assertEqual(module.layers[2].in_features, 2, Wrapper)

    def test_nested_lists(self):
        class Wrapper(DeeplayModule):
            def __init__(self, depth=3, width=3):
                super().__init__()
                self.layers = self.recursive(depth, width)

            def recursive(self, depth, width):
                if depth == 0:
                    return Layer(nn.Linear, 1, 1)
                else:
                    layerlist = LayerList()
                    for _ in range(width):
                        layerlist.append(self.recursive(depth - 1, width))
                    return layerlist

        list_33 = Wrapper(3, 3)
        list_33.build()
        self.assertEqual(len(list_33.layers), 3)
        for layer in list_33.layers:
            self.assertEqual(len(layer), 3)
            for _layer in layer:
                self.assertEqual(len(_layer), 3)
                for __layer in _layer:
                    self.assertIsInstance(__layer, nn.Linear)

    def test_nested_lists2(self):
        class Wrapper(DeeplayModule):
            def __init__(self, depth=3, width=3):
                super().__init__()
                self.layers: LayerList = self.recursive(depth, width)

            def recursive(self, depth, width):
                if depth == 0:
                    return Layer(nn.Linear, 1, 1)
                else:
                    return LayerList(
                        *[self.recursive(depth - 1, width) for i in range(width)]
                    )

        list_33 = Wrapper(3, 3)
        list_33.build()
        self.assertEqual(len(list_33.layers), 3)
        for layer in list_33.layers:
            self.assertEqual(len(layer), 3)
            for _layer in layer:
                self.assertEqual(len(_layer), 3)
                for __layer in _layer:
                    self.assertIsInstance(__layer, nn.Linear)

    def test_configure_nested_lists(self):
        class Wrapper(DeeplayModule):
            def __init__(self, depth=3, width=3):
                super().__init__()
                self.layers = self.recursive(depth, width)

            def recursive(self, depth, width):
                if depth == 0:
                    return Layer(nn.Linear, 1, 1)
                else:
                    return LayerList(
                        *[self.recursive(depth - 1, width) for i in range(width)]
                    )

        list_33 = Wrapper(3, 3)
        list_33.layers[0][0][0].configure(in_features=2)
        list_33.layers.configure(0, 0, 1, in_features=3)
        list_33.layers.configure(slice(1, 3), [1, 2], in_features=4)

        list_33.build()
        for i, j, k in itertools.product(range(3), range(3), range(3)):
            if i == 0 and j == 0 and k == 0:
                self.assertEqual(list_33.layers[i][j][k].in_features, 2)
            elif i == 0 and j == 0 and k == 1:
                self.assertEqual(list_33.layers[i][j][k].in_features, 3)
            elif i in [1, 2] and j in [1, 2]:
                self.assertEqual(list_33.layers[i][j][k].in_features, 4)
            else:
                self.assertEqual(list_33.layers[i][j][k].in_features, 1)

    def test_with_instantiated(self):
        llist = LayerList(nn.Linear(1, 1), nn.Linear(1, 1))
        llist.build()
        self.assertEqual(len(llist), 2)
        self.assertIsInstance(llist[0], nn.Linear)
        self.assertIsInstance(llist[1], nn.Linear)

    def test_with_instantiated_2(self):
        class Item(DeeplayModule):
            def __init__(self):
                self.net = Layer(nn.Linear, 1, 1)

        llist = LayerList(Item(), Item())

        nets = llist.net
        self.assertEqual(len(nets), 2)
        self.assertIsInstance(nets[0], Layer)
        self.assertIsInstance(nets[1], Layer)

        llist.build()
        nets = llist.net
        self.assertEqual(len(nets), 2)
        self.assertIsInstance(nets[0], nn.Linear)
        self.assertIsInstance(nets[1], nn.Linear)

    def test_slice_does_not_mutate(self):
        llist = LayerList(
            LayerActivation(Layer(nn.Linear, 1, 1), Layer(nn.ReLU)),
            LayerActivation(Layer(nn.Linear, 1, 1), Layer(nn.ReLU)),
            LayerActivation(Layer(nn.Linear, 1, 1), Layer(nn.ReLU)),
            LayerActivation(Layer(nn.Linear, 1, 1), Layer(nn.ReLU)),
        )
        llist.build()
        llist[0:1]
        for layer in llist:
            self.assertTrue(layer._has_built)
        for layer in llist[0:]:
            self.assertTrue(layer._has_built)

    def test_layer_list_append_after_init(self):
        llist = Wrapper1(3)
        llist.layers.append(Layer(nn.Linear, 1, 1))
        self.assertEqual(len(llist.layers), 4)
        llist.build()
        self.assertEqual(len(llist.layers), 4)

    def test_layer_list_extend_after_init(self):
        llist = Wrapper1(3)
        llist.layers.extend([Layer(nn.Linear, 1, 1), Layer(nn.Linear, 1, 1)])
        self.assertEqual(len(llist.layers), 5)
        llist.build()
        self.assertEqual(len(llist.layers), 5)

    def test_layer_list_insert_after_init(self):
        llist = Wrapper1(3)
        llist.layers.insert(1, Layer(nn.Tanh))
        self.assertEqual(len(llist.layers), 4)
        self.assertIs(llist.layers[1].classtype, nn.Tanh)
        llist.build()
        self.assertEqual(len(llist.layers), 4)
        self.assertIsInstance(llist.layers[1], nn.Tanh)

    def test_layer_list_pop_after_init(self):
        llist = Wrapper1(3)
        llist.layers.pop()
        self.assertEqual(len(llist.layers), 2)
        llist.build()
        self.assertEqual(len(llist.layers), 2)

    def test_set_mapping(self):
        class AggregationRelu(nn.Module):
            def forward(self, x, A):
                return nn.functional.relu(A @ x)

        llist = LayerList(
            LayerActivation(Layer(nn.Linear, 1, 16), Layer(AggregationRelu)),
            LayerActivation(Layer(nn.Linear, 16, 16), Layer(AggregationRelu)),
            LayerActivation(Layer(nn.Linear, 16, 1), Layer(AggregationRelu)),
        )
        llist.layer.set_input_map("x")
        llist.layer.set_output_map("x")

        llist.activation.set_input_map("x", "A")
        llist.activation.set_output_map("x")

        for layer in llist.layer:
            self.assertEqual(layer.input_args, ("x",))
            self.assertEqual(layer.output_args, {"x": 0})

        for activation in llist.activation:
            self.assertEqual(activation.input_args, ("x", "A"))
            self.assertEqual(activation.output_args, {"x": 0})

    def test_configuration_applies_in_wrapped(self):
        class MLP(DeeplayModule):
            def __init__(self, in_features, hidden_features, out_features):
                self.in_features = in_features
                self.hidden_features = hidden_features
                self.out_features = out_features

                self.blocks = LayerList()
                self.blocks.append(
                    LayerActivation(
                        Layer(nn.Linear, in_features, hidden_features[0]),
                        Layer(nn.ReLU),
                    )
                )

                self.hidden_blocks = LayerList()
                for i in range(len(hidden_features) - 1):
                    self.hidden_blocks.append(
                        LayerActivation(
                            Layer(
                                nn.Linear, hidden_features[i], hidden_features[i + 1]
                            ),
                            Layer(nn.ReLU),
                        )
                    )

                self.blocks.extend(self.hidden_blocks)
                self.blocks.insert(
                    1,
                    LayerActivation(
                        Layer(nn.Linear, hidden_features[-1], out_features),
                        Layer(nn.Sigmoid),
                    ),
                )

        class TestClass(DeeplayModule):
            def __init__(self, model=None):
                super().__init__()

                model = MLP(1, [1, 1], 1)
                model.blocks[0].layer.configure(out_features=2, in_features=2)
                model.blocks[1].layer.configure(in_features=2, out_features=2)
                model.blocks[2].layer.configure(in_features=2, out_features=2)
                self.model = model

        testclass = TestClass()
        testclass.build()
        self.assertEqual(len(testclass.model.blocks), 3)
        self.assertEqual(testclass.model.blocks[0].layer.in_features, 2)
        self.assertEqual(testclass.model.blocks[1].layer.in_features, 2)
        self.assertEqual(testclass.model.blocks[2].layer.in_features, 2)
        self.assertEqual(testclass.model.blocks[0].layer.out_features, 2)
        self.assertEqual(testclass.model.blocks[1].layer.out_features, 2)
        self.assertEqual(testclass.model.blocks[2].layer.out_features, 2)

    def test_configure_sequential_sub_model(self):
        from deeplay import ConvolutionalNeuralNetwork, MultiLayerPerceptron

        model = Sequential(
            ConvolutionalNeuralNetwork(1, [32, 64], 96),
            MultiLayerPerceptron(96, [128], 1),
        )
        model[1].configure(hidden_features=[128, 256, 512])
        model = model.create()

        self.assertEqual(len(model[1].blocks[1:-1]), 2)
        self.assertEqual(model[1].blocks[0].in_features, 96)
        self.assertEqual(model[1].blocks[0].out_features, 128)
        self.assertEqual(model[1].blocks[1].in_features, 128)
        self.assertEqual(model[1].blocks[1].out_features, 256)
        self.assertEqual(model[1].blocks[2].in_features, 256)
        self.assertEqual(model[1].blocks[2].out_features, 512)
        self.assertEqual(model[1].blocks[3].in_features, 512)
        self.assertEqual(model[1].blocks[3].out_features, 1)


class TestSequential(unittest.TestCase):
    def test_set_inp_out_mapping_1(self):
        model = Sequential(
            Layer(nn.Linear, 1, 20),
            Layer(nn.ReLU),
            Layer(nn.Linear, 20, 1),
        )
        model.set_input_map("x")
        model.set_output_map("x")

        model.build()

        inp = {"x": torch.randn(10, 1)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))

    def test_set_inp_out_mapping_2(self):
        model = Sequential(
            Layer(nn.Linear, 1, 20),
            Layer(nn.ReLU),
            Layer(nn.Linear, 20, 1),
        )
        model.set_input_map("x")
        model.set_output_map("x")

        model[1].set_output_map("x", x1=0, x2=0)

        model.build()

        inp = {"x": torch.randn(10, 1)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))
        self.assertEqual(torch.all(out["x1"] == out["x2"]), True)

    def test_forward_with_input_dict(self):
        class AggregationRelu(nn.Module):
            def forward(self, x, A):
                return nn.functional.relu(A @ x)

        model = Sequential(
            Layer(nn.Linear, 1, 20),
            Layer(AggregationRelu),
            Layer(nn.Linear, 20, 1),
        )

        model[0].set_input_map("x")
        model[0].set_output_map("x")

        model[1].set_input_map("x", "A")
        model[1].set_output_map("x")

        model[2].set_input_map("x")
        model[2].set_output_map("x")

        model.build()

        inp = {"x": torch.randn(10, 1), "A": torch.randn(10, 10)}
        out = model(inp)
        self.assertEqual(out["x"].shape, (10, 1))


class Module_1(DeeplayModule):
    def forward(self, x):
        return x, x * 2


class Module_2(nn.Module):
    def forward(self, x):
        return x / 2


class TestParallel(unittest.TestCase):
    def test_parallel_default(self):
        model = Parallel(Module_1(), Layer(Module_2))
        model.build()

        out = model(2.0)
        self.assertEqual(out[0], (2.0, 4.0))
        self.assertEqual(out[1], 1.0)

    def test_parallel_with_dict_inputs(self):
        model_1 = Module_1()
        model_1.set_input_map("x")
        model_1.set_output_map("x1", "x2")  # adds x1, x2 to output

        model_2 = Layer(Module_2)
        model_2.set_input_map("x")
        model_2.set_output_map("x3")  # adds x3 to output

        model = Parallel(model_1, model_2)
        model.build()

        inp = {"x": 2.0}
        out = model(inp)

        self.assertEqual(out["x"], 2.0)
        self.assertEqual(out["x1"], 2.0)
        self.assertEqual(out["x2"], 4.0)
        self.assertEqual(out["x3"], 1.0)

    def test_parallel_with_kwargs(self):
        model_1 = Module_1()
        model_1.set_input_map("x")
        model_1.set_output_map("x1", "x2")

        model_2 = Layer(Module_2)
        model_2.set_input_map("x")

        model = Parallel(model_1, x3=model_2)
        model.build()

        inp = {"x": 2.0}
        out = model(inp)

        self.assertEqual(out["x"], 2.0)
        self.assertEqual(out["x1"], 2.0)
        self.assertEqual(out["x2"], 4.0)
        self.assertEqual(out["x3"], 1.0)

    def test_parallel_with_kwargs_2(self):
        model_1 = Module_1()
        model_1.set_input_map("x")
        model_1.set_output_map("x1", "x2")

        model_2 = Layer(Module_2)
        model_2.set_input_map("x")
        model_2.set_output_map("x3", x4=0)

        model = Parallel(model_1, x5=model_2)
        model.build()

        inp = {"x": 2.0}
        out = model(inp)

        self.assertEqual(out["x"], 2.0)
        self.assertEqual(out["x1"], 2.0)
        self.assertEqual(out["x2"], 4.0)
        self.assertTrue("x3" not in out)
        self.assertTrue("x4" not in out)
        self.assertEqual(out["x5"], 1.0)


class TestReferringLayerList(unittest.TestCase):
    def test_referring_layer_list_from_LayerList(self):

        layerlist = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        referring = layerlist.layer

        self.assertIsInstance(referring, ReferringLayerList)
        self.assertEqual(len(referring), 2)
        self.assertIs(referring[0], layerlist[0].layer)
        self.assertIs(referring[1], layerlist[1].layer)

    def test_referring_layer_list_from_LayerList_method(self):

        layerlist = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        referring = layerlist.activated

        self.assertIsInstance(referring, ReferringLayerList)
        self.assertEqual(len(referring), 2)

        referring(nn.ReLU)

        self.assertIs(layerlist[0].activation.classtype, nn.ReLU)
        self.assertIs(layerlist[1].activation.classtype, nn.ReLU)

    def test_from_sliced_LayerList(self):

        layerlist = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        referring = layerlist[0:2].layer

        self.assertIsInstance(referring, ReferringLayerList)
        self.assertEqual(len(referring), 2)
        self.assertIs(referring[0], layerlist[0].layer)
        self.assertIs(referring[1], layerlist[1].layer)

    def test_from_sliced_LayerList_method(self):

        layerlist = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        referring = layerlist[0:2].activated

        self.assertIsInstance(referring, ReferringLayerList)
        self.assertEqual(len(referring), 2)

        referring(nn.ReLU)

        self.assertIs(layerlist[0].activation.classtype, nn.ReLU)
        self.assertIs(layerlist[1].activation.classtype, nn.ReLU)
        self.assertFalse(hasattr(layerlist[2], "activation"))

    def test_add_two_referring_layer_lists(self):

        layerlist_1 = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        layerlist_2 = LayerList(Conv2dBlock(1, 1), Conv2dBlock(1, 1))
        referring = layerlist_1.layer + layerlist_2.layer

        self.assertIsInstance(referring, ReferringLayerList)
        self.assertEqual(len(referring), 4)
        self.assertIs(referring[0], layerlist_1[0].layer)
        self.assertIs(referring[1], layerlist_1[1].layer)
        self.assertIs(referring[2], layerlist_2[0].layer)
        self.assertIs(referring[3], layerlist_2[1].layer)
