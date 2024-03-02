from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union

from ... import (
    DeeplayModule,
    Layer,
    LayerList,
    Sequential,
    LayerActivation,
    LayerActivationNormalization,
    PoolLayerActivationNormalization,
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


class DoubleConvBlock(DeeplayModule):
    """
    Connects two base blocks in series either with or without a skip connection. Acts like a residual block.
    """

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.blocks = LayerList()
        self.blocks.append(Block(in_channels, out_channels))
        self.blocks.append(Block(out_channels, out_channels, activation=nn.Identity()))
        self.residual = residual

    def forward(self, x):
        x_input = x
        for block in self.blocks:
            x = block(x)
        return F.gelu(x_input + x) if self.residual else x


class AttentionBlock(DeeplayModule):
    """
    Applies attention mechanism to the input tensor. Depending on the input, it can handle both self-attention and cross-attention.
    """

    def __init__(self, channels, context_embedding_dim):
        super().__init__()
        self.channels = channels

        # Self-attention part of the basic transformer
        self.layer_norm1 = Layer(nn.LayerNorm, [channels])
        self.self_attention = Layer(
            nn.MultiheadAttention, channels, num_heads=1, batch_first=True
        )

        # Cross-attention if context is enabled
        if context_embedding_dim is not None:
            self.cross_attention = Layer(
                nn.MultiheadAttention, channels, num_heads=1, batch_first=True
            )
            self.layer_norm2 = Layer(nn.LayerNorm, [channels])
            self.context_projection = Layer(nn.Linear, context_embedding_dim, channels)

        # Feedforward part of the basic transformer
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
        x_input = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = x + x_input

        # Reshape back to original shape: [B, H*W, C] -> [B, C, H, W]
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


class FeatureIntegrationModule(DeeplayModule):
    """
    Integrates the features with context information (such as time, text, etc.) using positional encoding and self-attention.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim,
        context_embedding_dim,
        enable_attention=True,
    ):
        super().__init__()
        self.conv_block = (
            Sequential(
                DoubleConvBlock(in_channels, in_channels, residual=True),
                DoubleConvBlock(in_channels, out_channels),
            )
            if enable_attention
            else DoubleConvBlock(in_channels, out_channels)
        )
        self.feed_forward_positional_embedding = Layer(
            nn.Linear, embedding_dim, out_channels
        )
        # self.attention_layer = (
        #     AttentionBlock(
        #         channels=out_channels, context_embedding_dim=context_embedding_dim
        #     )
        #     if enable_attention
        #     else Layer(nn.Identity)
        # )
        self.enable_attention = enable_attention
        if self.enable_attention:
            self.attention_layer = AttentionBlock(
                channels=out_channels, context_embedding_dim=context_embedding_dim
            )
            # self.use_attention = True
        # else:
        #     self.attention_layer = Layer(nn.Identity)
        #     self.use_attention = False

    def forward(self, x, t, context):
        x = self.conv_block(x)
        # Tranform the time step embedding to the channel dimension of the input feature map
        emb = self.feed_forward_positional_embedding(t)
        # Repeat and reshape the embedding to match the spatial dimensions of the input feature map
        emb = emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        # Add the positional encoding to the input feature map
        x = x + emb

        # Apply self-attention to the input feature map
        if self.enable_attention:
            x = self.attention_layer(x, context)

        return x


class UNetEncoder(DeeplayModule):
    """
    UNet encoder.

    The time step information is not integrated in the first block. It is still included in channel_attention just for the sake of consistency.
    """

    def __init__(
        self,
        in_channels,
        channels,
        channel_attention,
        position_embedding_dim,
        context_embedding_dim,
    ):
        super().__init__()
        self.conv_block1 = DoubleConvBlock(in_channels, channels[0])
        self.blocks = LayerList()

        for i in range(len(channels) - 1):
            attention_flag = channel_attention[i + 1]
            self.blocks.append(
                FeatureIntegrationModule(
                    channels[i],
                    channels[i + 1],
                    position_embedding_dim,
                    enable_attention=attention_flag,
                    context_embedding_dim=context_embedding_dim,
                )
            )
        self.pool = Layer(nn.MaxPool2d, kernel_size=2, stride=2)

    def forward(self, x, t, context):
        feature_maps = []
        x = self.conv_block1(x)
        feature_maps.append(x)
        x = self.pool(x)

        for block in self.blocks:
            x = block(x, t, context)
            feature_maps.append(x)
            x = self.pool(x)
        return feature_maps


class UNetDecoder(DeeplayModule):
    """
    UNet decoder.
    """

    def __init__(
        self, channels, channel_attention, position_embedding_dim, context_embedding_dim
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
                        enable_attention=attention_flag,
                        context_embedding_dim=context_embedding_dim,
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

    The first attention flag will be ignored as the time information is not integrated at this step. It is still included in the channel_attention just for the sake of consistency.

    If num_classes are provided, the class embedding will be added to the positional encoding. (This does not mean that the model will be ready for class conditional ddpm. You still have to classifier free guidance.)
    """

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
    ):
        super().__init__()
        self.position_embedding_dim = position_embedding_dim
        self.context_embedding_dim = context_embedding_dim

        # Class embedding
        if num_classes is not None:
            self.class_embedding = Layer(
                nn.Embedding, num_classes, position_embedding_dim
            )

        # Checks
        if len(channel_attention) != len(channels):
            raise ValueError(
                "The number of attention flags should be equal to the number of channels."
            )

        # UNet encoder
        self.encoder = UNetEncoder(
            in_channels,
            channels,
            channel_attention,
            position_embedding_dim,
            context_embedding_dim,
        )

        # Base blocks
        self.base_blocks = LayerList()
        self.base_blocks.append(DoubleConvBlock(channels[-1], base_channels[0]))
        for i in range(len(base_channels) - 1):
            self.base_blocks.append(
                DoubleConvBlock(base_channels[i], base_channels[i + 1])
            )
        self.base_blocks.append(DoubleConvBlock(base_channels[-1], channels[-1]))

        # UNet decoder
        self.decoder = UNetDecoder(
            channels, channel_attention, position_embedding_dim, context_embedding_dim
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
                    + "Please make sure that the embedding dimensions given to the model and the context dimension provided in forward function match."
                )

        if y is not None:
            y = self.class_embedding(y)
            t = t + y

        feature_maps = self.encoder(x, t, context)

        for block in self.base_blocks:
            feature_maps[-1] = block(feature_maps[-1])

        x = self.decoder(feature_maps[-1], feature_maps[:-1], t, context)
        x = self.output(x)
        return x
