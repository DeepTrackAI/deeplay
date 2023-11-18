import unittest
import torch
import torch.nn as nn
from deeplay import LayerList, DeeplayModule, Layer
import itertools


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
