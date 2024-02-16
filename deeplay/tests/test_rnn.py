import unittest
import torch
import torch.nn as nn
from deeplay import RecurrentNeuralNetwork, Layer


class TestComponentRNN(unittest.TestCase):
    ...

    def test_mlp_defaults(self):
        rnn = RecurrentNeuralNetwork(2, [4], 3)
        rnn.build()
        self.assertEqual(len(rnn.blocks), 2)

        self.assertEqual(rnn.blocks[0].layer.input_size, 2)
        self.assertEqual(rnn.blocks[0].layer.hidden_size, 4)

        self.assertEqual(rnn.output.layer.input_size, 4)
        self.assertEqual(rnn.output.layer.hidden_size, 3)

        # test on a batch of 2
        x = torch.randn(10, 1, 2)
        y = rnn(x)
        self.assertEqual(y.shape, (10, 1, 3))

    def test_mlp_change_depth(self):
        rnn = RecurrentNeuralNetwork(2, [4], 3)
        rnn.configure(hidden_features=[4, 4])
        rnn.build()
        self.assertEqual(len(rnn.blocks), 3)

    def test_bidirectional(self):
        rnn = RecurrentNeuralNetwork(2, [4], 3)
        rnn.blocks.layer.configure(bidirectional=True)
        rnn.build()
        self.assertEqual(len(rnn.blocks), 2)
        self.assertTrue(rnn.blocks[0].layer.bidirectional)

    def test_no_hidden_layers(self):
        rnn = RecurrentNeuralNetwork(2, [], 3)
        rnn.build()
        self.assertEqual(len(rnn.blocks), 1)
        self.assertEqual(rnn.blocks[0].layer.input_size, 2)
        self.assertEqual(rnn.blocks[0].layer.hidden_size, 3)
