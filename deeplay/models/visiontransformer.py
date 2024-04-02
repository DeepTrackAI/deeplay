from deeplay import (
    DeeplayModule,
    Layer,
    MultiLayerPerceptron,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

import torch
import torch.nn as nn

from typing import Optional, Sequence, Type, Union


class Patchify(DeeplayModule):
    """Patchify module.

    Splits an image into patches, linearly embeds them, and (optionally) applies dropout to the embeddings.

    Parameters
    ----------
    in_channels: int or None
        Number of input features. If None, the input shape is inferred from the first forward pass.
    out_features : int
        Number of output features.
    patch_size : int
        Size of the patch. The image is divided into patches of size `patch_size x patch_size` pixels.

    Constraints
    -----------
    - input_shape: (batch_size, in_channels, height, width)
    - output_shape: (batch_size, num_patches, out_features)

     Examples
    --------
    >>> embedder = Patchify(in_channels=3, out_features=256, patch_size=4)
    >>> # Customizing dropout
    >>> embedder.dropout.configure(p=0.1)

    Return Values
    -------------
    The forward method returns the processed tensor.

    """

    in_channels: Optional[int]
    out_features: int
    patch_size: int

    def __init__(self, in_channels: Optional[int], out_features: int, patch_size: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.patch_size = patch_size

        if out_features <= 0:
            raise ValueError(f"out_channels must be positive, got {out_features}")

        if in_channels is not None and in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.layer = (
            Layer(
                nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_features,
                kernel_size=patch_size,
                stride=patch_size,
            )
            if in_channels
            else Layer(
                nn.LazyConv2d,
                out_channels=out_features,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )
        self.dropout = Layer(nn.Dropout, p=0)

    def forward(self, x):
        x = self.layer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.dropout(x)
        return x


class ViT(DeeplayModule):
    """
    Vision Transformer (ViT) model.

    Parameters
    ----------
    image_size : int
        Size of the input image. The image is assumed to be square.
    patch_size : int
        Size of the patch. The image is divided into patches of size `patch_size x patch_size` pixels.
    in_channels : int or None
        Number of input channels. If None, the input shape is inferred from the first forward pass.
    hidden_features : Sequence[int]
        Number of hidden features for each layer of the transformer encoder.
    out_features : int
        Number of output features.
    num_heads : int
        Number of attention heads in multihead attention layers of the transformer encoder.
    out_activation: template-like or None
        Specification for the output activation of the model (Default: nn.Identity).

    Configurables
    -------------
    - in_channels (int): Number of input features. If None, the input shape is inferred from the first forward pass.
    - hidden_features (list[int]): Number of hidden units in each transformer layer.
    - out_features (int): Number of output features.
    - num_heads (int): Number of attention heads in multihead attention layers.
    - patch_embedder (template-like): Specification for the patch embedder (Default: dl.Patchify).
    - positional_embedder (template-like): Specification for the positional embedder (Default: dl.PositionalEmbedding).
    - transformer_encoder (template-like): Specification for the transformer encoder layer (Default: dl.TransformerEncoderLayer).
    - dense_top (template-like): Specification for the dense top layer (Default: dl.MultiLayerPerceptron).

    Constraints
    -----------
    - input_shape: (batch_size, in_channels, image_size, image_size)
    - output_shape: (batch_size, out_features)

    Examples
    --------
    >>> vit = ViT(
    >>>       image_size=32,
    >>>       patch_size=4,
    >>>       hidden_features=[384,] * 7,
    >>>       out_channels=10,
    >>>       num_heads=12,
    >>> ).create()
    >>> # Testing on a batch of 2
    >>> x = torch.randn(2, 3, 32, 32)
    >>> vit(x).shape
    torch.Size([2, 10])

    Return Values
    -------------
    The forward method returns the processed tensor.

    """

    in_channels: Optional[int]
    image_size: int
    patch_size: int
    hidden_features: Sequence[Optional[int]]
    out_features: int

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
        in_channels: Optional[int],
        image_size: int,
        patch_size: int,
        hidden_features: Sequence[int],
        out_features: int,
        num_heads: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_features = hidden_features
        self.out_channels = out_features
        self.num_heads = num_heads

        assert (
            image_size % patch_size == 0
        ), f"image_size must be divisible by patch_size. Found {image_size} and {patch_size}."
        num_patches = (image_size // patch_size) ** 2

        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        if any(h <= 0 for h in hidden_features):
            raise ValueError(f"hidden_features must be positive, got {hidden_features}")

        self.patch_embedder = Patchify(
            in_channels=in_channels,
            out_features=hidden_features[0],
            patch_size=patch_size,
        )

        # Initialize cls token as a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_features[0]))

        # Positional embeddings. We add 1 to the number of patches to account for the cls token.
        # The positional embeddings are learnable and initialized with normal distribution.
        # batch_first is set to True to match the constraints of the model.
        self.positional_embedder = PositionalEmbedding(
            features=hidden_features[0],
            max_length=1 + num_patches,
            learnable=True,
            initializer=torch.nn.init.normal_,
            batch_first=True,
        )

        # Transformer encoder.
        self.transformer_encoder = TransformerEncoderLayer(
            in_features=hidden_features[0],
            hidden_features=hidden_features[:-1],
            out_features=hidden_features[-1],
            num_heads=num_heads,
            batch_first=True,
        )
        # Follow the order: normalization -> layer -> dropout -> skip as in the original paper.
        self.transformer_encoder[..., "multihead|feed_forward"].configure(
            order=["normalization", "layer", "dropout", "skip"]
        )
        # All transformer layers use GELU activation by default
        self.transformer_encoder[..., "activation"].configure(nn.GELU)

        # Dense top layer. The output cls_token is passed through a MLP to get the final output.
        self.dense_top = MultiLayerPerceptron(
            in_features=hidden_features[-1],
            hidden_features=[hidden_features[-1] // 2, hidden_features[-1] // 4],
            out_features=out_features,
            out_activation=out_activation,
        )
        # Follow the order: normalization -> layer -> activation -> dropout as in the original paper.
        self.dense_top.blocks.configure(
            order=["normalization", "layer", "activation", "dropout"]
        )
        # All dense layers use GELU activation by default, except the last one.
        self.dense_top[..., "activation#:-1"].configure(nn.GELU)

    def forward(self, x):
        # Split the image into patches and embed them
        x = self.patch_embedder(x)

        # Add cls token to patches
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings to patches
        x = self.positional_embedder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Map hidden channels to output space
        x = self.dense_top(x[:, 0])

        return x
