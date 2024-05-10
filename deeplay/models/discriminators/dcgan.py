import torch
import torch.nn as nn

from deeplay.blocks.sequential import SequentialBlock
from deeplay.components import ConvolutionalEncoder2d
from deeplay.external.layer import Layer
from deeplay.initializers.normal import Normal


@ConvolutionalEncoder2d.register_style
def dcgan_discriminator(encoder: ConvolutionalEncoder2d):
    encoder.blocks.configure("layer", kernel_size=4, stride=2, padding=1)
    encoder.blocks[-1].configure("layer", padding=0)
    encoder["blocks", :].all.remove("pool", allow_missing=True)
    encoder["blocks", 1:-1].all.normalized()
    encoder["blocks", :-1].all.configure("activation", nn.LeakyReLU, negative_slope=0.2)
    encoder.blocks[-1].activation.configure(nn.Sigmoid)

    initializer = Normal(
        targets=(
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.Embedding,
            nn.Linear,
        ),
        mean=0,
        std=0.02,
    )
    encoder.initialize(initializer, tensors="weight")


class DCGANDiscriminator(ConvolutionalEncoder2d):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) discriminator.

    Parameters
    ----------
    input_channels: int
        Number of input channels
    features_dim: int
        Dimension of the features. The number of features in the four ConvBlocks of the Discriminator can be controlled by this parameter. Convolutional layers = [features_dim, features_dim*2, features_dim*4, features_dim*8].
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
    - input shape: (batch_size, ch_in, 64, 64)
    - output shape: (batch_size, 1, 1, 1)

    Examples
    --------
    >>> discriminator = DCGAN_Discriminator(input_channels=1, class_conditioned_model=False)
    >>> discriminator.build()
    >>> batch_size = 16
    >>> input = torch.randn(batch_size, 1, 64, 64)
    >>> output = discriminator(input)

    Return Values
    -------------
    The forward method returns the processed tensor.


    """

    input_channels: int
    class_conditioned_model: bool
    embedding_dim: int
    num_classes: int

    def __init__(
        self,
        in_channels: int = 1,
        features_dim: int = 64,
        class_conditioned_model: bool = False,
        embedding_dim: int = 100,
        num_classes: int = 10,
        input_channels=None,
    ):
        if input_channels is not None:
            in_channels = input_channels
        if class_conditioned_model:
            in_channels += 1

        self.in_channels = in_channels
        self.features_dim = features_dim
        self.class_conditioned_model = class_conditioned_model
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        super().__init__(
            in_channels=in_channels,
            hidden_channels=[
                features_dim,
                features_dim * 2,
                features_dim * 4,
                features_dim * 8,
            ],
            out_channels=1,
            out_activation=Layer(nn.Sigmoid),
        )
        self.features_dim = features_dim
        self.class_conditioned_model = class_conditioned_model

        if class_conditioned_model:
            # TODO: prepend first block with label embedding
            # Needs ops (reshape) to match the input shape

            self.label_embedding = SequentialBlock(
                embedding=Layer(nn.Embedding, num_classes, embedding_dim),
                layer=Layer(nn.Linear, embedding_dim, 64 * 64),
                activation=Layer(nn.LeakyReLU, 0.2),
            )
        else:
            self.label_embedding = Layer(nn.Identity)

        self.style("dcgan_discriminator")

    def forward(self, x, y=None):
        expected_shape = (x.shape[0], self.in_channels, 64, 64)
        if x.shape[-2:] != expected_shape[-2:]:
            raise ValueError(
                f"Input shape is {x.shape}, expected {expected_shape}. DCGAN discriminator expects 64x64 images. "
            )

        if self.class_conditioned_model:
            if y is None:
                raise ValueError(
                    "Class label y must be provided for class-conditional discriminator"
                )

            y = self.label_embedding(y)
            y = y.view(-1, 1, 64, 64)
            x = torch.cat([x, y], dim=1)

        for block in self.blocks:
            x = block(x)

        return x
