from deeplay import (
    DeeplayModule,
    Layer,
    LayerList,
    MultiLayerPerceptron,
    PositionalEmbedding,
    TransformerEncoderLayer,
    Sequential,
)

import torch
import torch.nn as nn

from typing import Optional, Sequence, Type, Union


class Patchify(DeeplayModule):
    def __init__(self, features, patch_size):
        super().__init__()
        self.features = features
        self.patch_size = patch_size

        self.layer = Layer(
            nn.LazyConv2d,
            out_channels=features,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.dropout = Layer(nn.Dropout, p=0)

    def forward(self, x):
        x = self.layer(x).flatten(2).transpose(1, 2)
        x = self.dropout(x)
        return x


class ViT(DeeplayModule):
    """
    Vision Transformer (ViT) model.
    """

    image_size: int
    patch_size: int
    hidden_channels: Sequence[Optional[int]]
    out_channels: int
    blocks: LayerList[DeeplayModule]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `.patch_embedder`."""
        return self.patch_embedder

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `.transformer_encoder`."""
        return self.transformer_encoder

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `.dense_top`."""
        return self.dense_top

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        num_heads: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):

        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        assert (
            image_size % patch_size == 0
        ), f"image_size must be divisible by patch_size. Found {image_size} and {patch_size}."
        num_patches = (image_size // patch_size) ** 2

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if any(h <= 0 for h in hidden_channels):
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")

        self.patch_embedder = Patchify(
            features=hidden_channels[0],
            patch_size=patch_size,
        )

        # Initialize cls token as learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels[0]))

        # Add positional embeddings to patches. 1 + since it includes
        # the "position" of the cls token
        self.positional_embedder = PositionalEmbedding(
            features=hidden_channels[0],
            max_length=1 + num_patches,
            learnable=True,
            initializer=torch.nn.init.normal_,
            batch_first=True,
        )

        # Transformer encoder
        self.transformer_encoder = TransformerEncoderLayer(
            in_features=hidden_channels[0],
            hidden_features=hidden_channels[:-1],
            out_features=hidden_channels[-1],
            num_heads=num_heads,
            batch_first=True,
        )
        # processing order: normalization -> layer -> dropout -> skip
        self.transformer_encoder[..., "multihead|feed_forward"].configure(
            order=["normalization", "layer", "dropout", "skip"]
        )
        # GELU activation by default
        self.transformer_encoder[..., "activation"].configure(nn.GELU)

        # Dense top
        self.dense_top = MultiLayerPerceptron(
            in_features=hidden_channels[-1],
            hidden_features=[hidden_channels[-1] // 2, hidden_channels[-1] // 4],
            out_features=out_channels,
            out_activation=out_activation,
        )
        # processing order: normalization -> layer -> activation -> dropout
        self.dense_top.blocks.configure(
            order=["normalization", "layer", "activation", "dropout"]
        )
        # GELU activation by default
        self.dense_top[..., "activation#:-1"].configure(nn.GELU)

    def forward(self, x):
        # Patchify image into patches
        x = self.patch_embedder(x)

        # Add cls token to patches
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings to patches
        x = self.positional_embedder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Map hidden channels to output space
        x = self.dense_top(x[:, 0])

        return x
