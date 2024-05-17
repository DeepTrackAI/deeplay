import torch
import torch.nn as nn
from deeplay.blocks.conv.conv2d import Conv2dBlock
from deeplay.components import ConvolutionalDecoder2d
from deeplay.external.layer import Layer
from deeplay.initializers.normal import Normal


@ConvolutionalDecoder2d.register_style
def dcgan_generator(generator: ConvolutionalDecoder2d):
    generator.normalized(after_last_layer=False)
    generator[...].isinstance(Conv2dBlock).hasattr("layer").all.configure(
        "layer", nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1
    ).remove("upsample", allow_missing=True)
    generator.blocks[0].layer.configure(stride=1, padding=0)

    initializer = Normal(
        targets=(
            nn.ConvTranspose2d,
            nn.BatchNorm2d,
            nn.Embedding,
        ),
        mean=0,
        std=0.02,
    )
    generator.initialize(initializer, tensors="weight")


class DCGANGenerator(ConvolutionalDecoder2d):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) generator.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space
    feature_dims: int
        Dimension of the features. The number of features in the four ConvTransposeBlocks of the Generator can be controlled by this parameter. Convolutional transpose layers = [features_dim*16, features_dim*8, features_dim*4, features_dim*2].
    output_channels: int
        Number of output channels
    class_conditioned_model: bool
        Whether the model is class-conditional
    embedding_dim: int
        Dimension of the label embedding
    num_classes: int
        Number of classes

    Shorthands
    ----------
    - input: `.blocks[0]`
    - hidden: `.blocks[:-1]`
    - output: `.blocks[-1]`
    - layer: `.blocks.layer`
    - activation: `.blocks.activation`

    Constraints
    -----------
    - input shape: (batch_size, latent_dim)
    - output shape: (batch_size, ch_out, 64, 64)

    Examples
    --------
    >>> generator = DCGANGenerator(latent_dim=100, output_channels=1, class_conditioned_model=False)
    >>> generator.build()
    >>> batch_size = 16
    >>> input = torch.randn([batch_size, 100, 1, 1])
    >>> output = generator(x=input, y=None)

    Return Values
    -------------
    The forward method returns the processed tensor.

    """

    latent_dim: int
    output_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int

    def __init__(
        self,
        latent_dim: int = 100,
        features_dim: int = 64,
        out_channels: int = 1,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
        output_channels=None,
    ):
        if output_channels is not None:
            out_channels = output_channels

        self.latent_dim = latent_dim
        self.output_channels = out_channels
        self.class_conditioned_model = class_conditioned_model
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        in_channels = latent_dim
        if class_conditioned_model:
            in_channels += embedding_dim

        super().__init__(
            in_channels=in_channels,
            hidden_channels=[
                features_dim * 16,
                features_dim * 8,
                features_dim * 4,
                features_dim * 2,
            ],
            out_channels=out_channels,
            out_activation=Layer(nn.Tanh),
        )

        if class_conditioned_model:
            self.label_embedding = Layer(nn.Embedding, num_classes, embedding_dim)
        else:
            self.label_embedding = Layer(nn.Identity)

        self.style("dcgan_generator")

    def forward(self, x, y=None):
        if self.class_conditioned_model:
            if y is None:
                raise ValueError(
                    "Class label y must be provided for class-conditional generator"
                )

            y = self.label_embedding(y)
            y = y.view(-1, self.embedding_dim, 1, 1)
            x = torch.cat([x, y], dim=1)

        for block in self.blocks:
            x = block(x)

        return x
