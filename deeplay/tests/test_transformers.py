import unittest
import torch
import torch.nn as nn
from deeplay import TransformerEncoderLayer, LayerDropoutSkipNormalization, Add, ViT


class TestComponentTransformerEncoder(unittest.TestCase):
    ...

    def test_tel_defaults(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 2)

        for i in range(2):
            self.assertEqual(tel.blocks[i].multihead.layer.attention.embed_dim, 4)
            self.assertEqual(tel.blocks[i].multihead.layer.attention.num_heads, 2)

            self.assertEqual(tel.blocks[i].feed_forward.layer.layer[0].in_features, 4)
            self.assertEqual(tel.blocks[i].feed_forward.layer.layer[-1].out_features, 4)

        # test on a batch of 2
        x = torch.randn(10, 2, 4)
        y = tel(x)
        self.assertEqual(y.shape, (10, 2, 4))

    def test_tel_change_depth(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.configure(hidden_features=[4, 4])
        tel.build()
        self.assertEqual(len(tel.blocks), 3)

    def test_no_hidden_layers(self):
        tel = TransformerEncoderLayer(4, [], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 1)

        self.assertEqual(tel.blocks[0].multihead.layer.attention.embed_dim, 4)
        self.assertEqual(tel.blocks[0].multihead.layer.attention.num_heads, 2)

        self.assertEqual(tel.blocks[0].feed_forward.layer.layer[0].in_features, 4)
        self.assertEqual(tel.blocks[0].feed_forward.layer.layer[-1].out_features, 4)

    def test_variable_hidden_layers(self):
        tel = TransformerEncoderLayer(4, [4, 8, 16], 4, 2)
        tel.build()
        self.assertEqual(len(tel.blocks), 4)

        self.assertEqual(tel.blocks[0].multihead.layer.attention.embed_dim, 4)
        self.assertEqual(tel.blocks[1].multihead.layer.attention.embed_dim, 8)
        self.assertEqual(tel.blocks[2].multihead.layer.attention.embed_dim, 16)
        self.assertEqual(tel.blocks[3].multihead.layer.attention.embed_dim, 4)

    def test_tel_multihead_subcomponents(self):
        tel = TransformerEncoderLayer(4, [4], 4, 2)
        tel.multihead[0].layer.configure("return_attn", True)
        tel.build()

        multihead = tel.multihead[0].layer

        # We now evaluate an input tensor of shape (10, 4) with batch index
        # (0, 0, 0, 0, 0, 0, 0, 0, 0, 1). The last element of the batch index
        # indicates that the last element of the input tensor is not allowed
        # to attend to any other element.
        x = torch.randn(10, 4)
        batch_index = torch.zeros(10, dtype=torch.long)
        batch_index[-1] = 1

        y, attn = multihead(x, batch_index=batch_index)
        self.assertEqual(y.shape, (10, 4))
        self.assertEqual(attn.shape, (10, 10))

        self.assertEqual(attn.sum(dim=-1).sum(), 10)
        self.assertEqual(attn[-1, -1], 1.0)

        skip = tel.multihead[0].skip
        y1 = skip(y, x)  # as given by a dict mapping
        y2 = skip((y, x))

        self.assertEqual(y1.shape, (10, 4))
        self.assertEqual(y2.shape, (10, 4))
        self.assertEqual((y1 - y2).sum(), 0.0)

    def test_tel_skip_position(self):
        class test_module(nn.Module):
            def forward(self, x):
                return x * 2

        tel = LayerDropoutSkipNormalization(
            layer=test_module(),
            dropout=nn.Identity(),
            skip=Add(),
            normalization=nn.Identity(),
            order=["layer", "skip", "dropout", "normalization"],
        )

        x = torch.Tensor([2.0])
        y = tel(x)

        self.assertEqual(y, 6.0)

        tel = LayerDropoutSkipNormalization(
            layer=test_module(),
            dropout=nn.Identity(),
            skip=Add(),
            normalization=nn.Identity(),
            order=["skip", "layer", "dropout", "normalization"],
        )

        y = tel(x)
        self.assertEqual(y, 8.0)


class TestComponentViT(unittest.TestCase):
    ...

    def test_vit_defaults(self):
        vit = ViT(
            in_channels=3,
            image_size=32,
            patch_size=4,
            hidden_features=[
                384,
            ]
            * 7,
            out_features=10,
            num_heads=12,
        )
        vit.build()
        vit.create()

        self.assertEqual(len(vit.hidden.blocks), 7)

        self.assertEqual(vit.input.layer.in_channels, 3)
        self.assertEqual(vit.input.layer.out_channels, 384)

        self.assertEqual(vit.output.blocks[0].layer.in_features, 384)
        self.assertEqual(vit.output.blocks[-1].layer.out_features, 10)

        # test on a batch of 2
        x = torch.randn(2, 3, 32, 32)
        y = vit(x)
        self.assertEqual(y.shape, (2, 10))

    def test_vit_change_depth(self):
        vit = ViT(
            in_channels=3,
            image_size=32,
            patch_size=4,
            hidden_features=[
                384,
            ]
            * 7,
            out_features=10,
            num_heads=12,
        )
        vit.configure(hidden_features=[384, 384])
        vit.create()
        vit.build()
        self.assertEqual(len(vit.hidden.blocks), 2)

    def test_empty_hidden_features(self):
        vit = ViT(
            in_channels=3,
            image_size=32,
            patch_size=4,
            hidden_features=[
                384,
            ]
            * 7,
            out_features=10,
            num_heads=12,
        ).build()
        self.assertEqual(vit.input.layer.in_channels, 3)
        self.assertEqual(vit.input.layer.out_channels, 384)

        self.assertEqual(vit.output.blocks[0].layer.in_features, 384)
        self.assertEqual(vit.output.blocks[-1].layer.out_features, 10)

    def test_lazy_input(self):
        vit = ViT(
            None,
            image_size=32,
            patch_size=4,
            hidden_features=[
                384,
            ]
            * 7,
            out_features=10,
            num_heads=12,
        ).build()
        self.assertEqual(vit.input.layer.in_channels, 0)
        self.assertEqual(vit.input.layer.out_channels, 384)

        self.assertEqual(vit.output.blocks[0].layer.in_features, 384)
        self.assertEqual(vit.output.blocks[-1].layer.out_features, 10)

        # test on a batch of 2
        x = torch.randn(2, 3, 32, 32)
        y = vit(x)
        self.assertEqual(y.shape, (2, 10))
