from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from deeplay import (
    DeeplayModule,
    Layer,
    LayerList,
    Sequential,
    LayerActivationNormalization,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(LayerActivationNormalization):
    """
    Base block of the attention UNet. It consists of a convolutional layer, a group normalization layer, and a GELU activation layer.
    """

    def __init__(
        self, in_channels, out_channels, activation=None, normalization=None, **kwargs
    ):
        super().__init__(
            layer=Layer(nn.Conv2d, in_channels, out_channels, kernel_size=3, padding=1),
            activation=activation or Layer(nn.GELU),
            normalization=normalization or Layer(nn.GroupNorm, 1, out_channels),
            order=["layer", "normalization", "activation"],
        )


class AttentionBlock(DeeplayModule):
    """
    Applies attention mechanism to the input tensor. Depending on the input, it can handle both self-attention and cross-attention mechanisms. If context_embedding_dim is provided, it will apply cross-attention, else it will apply self-attention.
    """

    def __init__(self, channels, context_embedding_dim, num_attention_heads):
        super().__init__()
        self.channels = channels

        # Self-attention part of the basic transformer action
        self.layer_norm1 = Layer(nn.LayerNorm, [channels])
        self.self_attention = Layer(
            nn.MultiheadAttention,
            channels,
            num_heads=num_attention_heads["self"],
            batch_first=True,
        )

        # Cross-attention if context is enabled
        if context_embedding_dim is not None:
            self.cross_attention = Layer(
                nn.MultiheadAttention,
                channels,
                num_heads=num_attention_heads["cross"],
                batch_first=True,
            )
            self.layer_norm2 = Layer(nn.LayerNorm, [channels])
            self.context_projection = Layer(nn.Linear, context_embedding_dim, channels)

        # Feedforward part of the basic transformer action
        self.layer_norm3 = Layer(nn.LayerNorm, [channels])
        self.feed_forward = Sequential(
            Layer(nn.Linear, channels, channels),
            Layer(nn.GELU),
            Layer(nn.Linear, channels, channels),
        )

    def forward(self, x, context):
        # Reshape for multihead attention: [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Self-attention
        x = self.layer_norm1(x)
        self_attention_output, _ = self.self_attention(x, x, x)
        x = x + self_attention_output

        # Cross-attention if context is enabled
        if context is not None:
            context = self.context_projection(context)
            cross_attention_output, _ = self.cross_attention(
                self.layer_norm2(x), context, context
            )
            x = x + cross_attention_output

        # Feedforward
        z = self.layer_norm3(x)
        z = self.feed_forward(z)
        x = x + z

        # Reshape back to original shape: [B, H*W, C] -> [B, C, H, W]
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


class FeatureIntegrationModule(DeeplayModule):
    """
    Integrates the time and context information to the feature maps through residual connections and attention mechanisms.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        position_embedding_dim,
        context_embedding_dim,
        num_attention_heads,
        enable_attention=True,
    ):
        super().__init__()

        self.blocks = LayerList()
        self.blocks.append(Block(in_channels, out_channels))
        self.blocks.append(Block(out_channels, out_channels))

        # self.res_block = Block(in_channels, out_channels)
        self.res_block = Layer(nn.Conv2d, in_channels, out_channels, kernel_size=1)

        self.feed_forward_position_embedding = Layer(
            nn.Linear, position_embedding_dim, out_channels
        )

        self.enable_attention = enable_attention
        if self.enable_attention:
            self.attention_layer = AttentionBlock(
                channels=out_channels,
                context_embedding_dim=context_embedding_dim,
                num_attention_heads=num_attention_heads,
            )

    def forward(self, x, t, context):
        h = self.blocks[0](x)

        # Project the positional encoding and add it to the input feature map
        emb = self.feed_forward_position_embedding(t)
        h += emb[:, :, None, None]
        h = self.blocks[1](h)

        # Residual connection to the input feature map
        h += self.res_block(x)

        # Apply self-attention if enabled
        if self.enable_attention:
            h = self.attention_layer(h, context)
        return h


class UNetEncoder(DeeplayModule):
    """
    UNet encoder.

    Combines the double convolution blocks and the feature integration modules to create the encoder part of the UNet.
    """

    def __init__(
        self,
        in_channels,
        channels,
        channel_attention,
        position_embedding_dim,
        context_embedding_dim,
        num_attention_heads,
    ):
        super().__init__()
        self.blocks = LayerList()

        for i in range(len(channels)):
            attention_flag = channel_attention[i]
            self.blocks.append(
                FeatureIntegrationModule(
                    in_channels if i == 0 else channels[i - 1],
                    channels[i],
                    position_embedding_dim,
                    context_embedding_dim,
                    num_attention_heads,
                    enable_attention=attention_flag,
                )
            )

        self.pool = Layer(nn.MaxPool2d, kernel_size=2, stride=2)

    def forward(self, x, t, context):
        feature_maps = []
        for block in self.blocks:
            x = block(x, t, context)
            feature_maps.append(x)
            x = self.pool(x)
        return feature_maps


class UNetDecoder(DeeplayModule):
    """
    UNet decoder.

    Combines the convolutional transpose layers and the feature integration modules to create the decoder part of the UNet.
    """

    def __init__(
        self,
        channels,
        channel_attention,
        position_embedding_dim,
        context_embedding_dim,
        num_attention_heads,
    ):
        super().__init__()

        channels = channels[::-1]  # reverse the channels
        channel_attention = channel_attention[::-1]  # reverse the attention flags

        self.blocks = LayerList()

        for i in range(len(channels) - 1):
            attention_flag = channel_attention[i + 1]
            self.blocks.append(
                LayerList(
                    Layer(
                        nn.ConvTranspose2d,
                        channels[i],
                        channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    FeatureIntegrationModule(
                        channels[i],
                        channels[i + 1],
                        position_embedding_dim,
                        context_embedding_dim,
                        num_attention_heads,
                        enable_attention=attention_flag,
                    ),
                )
            )

    def forward(self, x, feature_maps, t, context):
        feature_maps = feature_maps[::-1]  # reverse the feature maps
        for i, block in enumerate(self.blocks):
            x = block[0](x)
            x = torch.cat([x, feature_maps[i]], dim=1)
            x = block[1](x, t, context)
        return x


class AttentionUNet(DeeplayModule):
    """
    Attention UNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : List[int]
        Number of channels in the encoder and decoder blocks.
    channel_attention : List[bool]
        Attention flags for the encoder and decoder blocks. If True, attention will be applied to the corresponding block. The first attention flag will be ignored as the time information is not integrated at this step. It is still included in the channel_attention just for the sake of consistency.
    base_channels : List[int]
        Number of channels in the base blocks.
    out_channels : int
        Number of output channels.
    position_embedding_dim : int
        Dimension of the positional encoding. Positional encoding is defined outside the model and passed as an input to the model. The dimension of the positional encoding should match the dimension given to the model.
    num_classes : Optional[int]
        Number of classes. If num_classes are provided, the class embedding will be added to the positional encoding. This is used for the class conditioned models.
    context_embedding_dim : Optional[int]
        Dimension of the context embedding. Context embedding is defined outside the model and passed as an input to the model. The dimension of the context embedding should match the dimension given to the model. When enabled, the context embedding will be used to apply cross-attention to the feature maps.
    num_attention_heads : dict
        Number of attention heads for self-attention and cross-attention mechanisms. The keys should be "self" and "cross" respectively. Default is {"self": 1, "cross": 1}.
    """

    in_channels: int
    channels: List[int]
    channel_attention: List[bool]
    base_channels: List[int]
    out_channels: int
    position_embedding_dim: int
    num_classes: Optional[int]
    context_embedding_dim: Optional[int]
    num_attention_heads: dict

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128],
        channel_attention: List[bool] = [True, True, True],
        base_channels: List[int] = [256, 256],
        out_channels: int = 1,
        position_embedding_dim: int = 16,
        num_classes: Optional[int] = None,
        context_embedding_dim: Optional[int] = None,
        num_attention_heads: dict = {"self": 1, "cross": 1},
    ):
        super().__init__()
        self.position_embedding_dim = position_embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.num_attention_heads = num_attention_heads

        # Class embedding
        if num_classes is not None:
            self.class_embedding = Layer(
                nn.Embedding, num_classes, position_embedding_dim
            )

        # Checks
        if len(channel_attention) != len(channels):
            raise ValueError(
                "Length of channel_attention should be equal to the length of channels"
            )

        # UNet encoder
        self.encoder = UNetEncoder(
            in_channels,
            channels,
            channel_attention,
            position_embedding_dim,
            context_embedding_dim,
            num_attention_heads,
        )

        self.base_blocks = LayerList()
        self.base_blocks.append(Block(channels[-1], base_channels[0]))
        for i in range(len(base_channels) - 1):
            self.base_blocks.append(Block(base_channels[i], base_channels[i + 1]))
        self.base_blocks.append(Block(base_channels[-1], channels[-1]))

        # UNet decoder
        self.decoder = UNetDecoder(
            channels,
            channel_attention,
            position_embedding_dim,
            context_embedding_dim,
            num_attention_heads,
        )

        # Output layer
        self.output = Layer(nn.Conv2d, channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, y=None, context=None):

        if t.shape[1] == 1:
            raise ValueError(
                "Time steps should be passed through a positional encoding function before passing it to the model."
            )

        if t.shape[1] != self.position_embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch. "
                + f"Expected: {self.position_embedding_dim}, Got: {t.shape[1]}. "
                + "Please make sure that the embedding dimensions given to the model and the positional encoding function match."
            )

        if context is not None:
            if context.shape[-1] != self.context_embedding_dim:
                raise ValueError(
                    "Embedding dimension mismatch. "
                    + f"Expected: {self.context_embedding_dim}, Got: {context.shape[2]}. "
                    + "Please make sure that the context embedding dimensions provided while instantiating the model and the context embedding dimensions match."
                )

        if y is not None:
            y = self.class_embedding(y)
            t += y

        feature_maps = self.encoder(x, t, context)

        for block in self.base_blocks:
            feature_maps[-1] = block(feature_maps[-1])

        x = self.decoder(feature_maps[-1], feature_maps[:-1], t, context)
        x = self.output(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128],
        channel_attention: List[bool] = [True, True, True],
        base_channels: List[int] = [256, 256],
        out_channels: int = 1,
        position_embedding_dim: int = 256,
        num_classes: Optional[int] = None,
        context_embedding_dim: Optional[int] = None,
        num_attention_heads: dict = {"self": 1, "cross": 1},
    ) -> None: ...

    configure = DeeplayModule.configure
