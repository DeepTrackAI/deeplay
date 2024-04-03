class CycleGANGenerator(DeeplayModule):
    """
    CycleGAN generator.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    n_residual_blocks : int
        Number of residual blocks in the generator.

    Shorthands
    ----------
    - input: `.blocks[0]`
    - hidden: `.blocks[:-1]`
    - output: `.blocks[-1]`
    - layer: `.blocks.layer`
    - activation: `.blocks.activation`

    Examples
    --------
    >>> generator = CycleGANGenerator(in_channels=1, out_channels=3)
    >>> generator.build()
    >>> x = torch.randn(1, 1, 256, 256)
    >>> y = generator(x)
    >>> y.shape

    Return values
    -------------
    The forward method returns the processed tensor.

    """

    in_channels: int
    out_channels: int
    n_residual_blocks: int
    blocks: LayerList[Layer]

    @property
    def input(self):
        """Return the input layer of the network. Equivalent to `self.blocks[0]`."""
        return self.blocks[0]

    @property
    def hidden(self):
        """Return the hidden layers of the network. Equivalent to `self.blocks[:-1]`."""
        return self.blocks[:-1]

    @property
    def output(self):
        """Return the last layer of the network. Equivalent to `self.blocks[-1]`."""
        return self.blocks[-1]

    @property
    def activation(self) -> LayerList[Layer]:
        """Return the activations of the network. Equivalent to `.blocks.activation`."""
        return self.blocks.activation

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 9,
    ):
        super().__init__()

        self.blocks = LayerList()

        # Initial convolution block
        self.blocks.append(
            CycleGANBlock(in_channels, 64, kernel_size=7, stride=1, padding=3)
        )

        # Downsampling convolutions
        self.blocks.append(CycleGANBlock(64, 128, kernel_size=3, stride=2, padding=1))
        self.blocks.append(CycleGANBlock(128, 256, kernel_size=3, stride=2, padding=1))

        # Residual blocks
        for _ in range(n_residual_blocks):
            self.blocks.append(ResidualBlock(channels=256))

        # Upsampling convolutions
        self.blocks.append(
            CycleGANBlock(
                256,
                128,
                transposed_conv=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        self.blocks.append(
            CycleGANBlock(
                128,
                64,
                transposed_conv=True,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )

        # Output layer
        self.blocks.append(
            CycleGANBlock(
                64,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                activation=nn.Tanh(),
                normalization=nn.Identity(),
            )
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @overload
    def configure(
        self,
        /,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 9,
    ) -> None: ...

    @overload
    def configure(
        self,
        name: Literal["blocks"],
        order: Optional[Sequence[str]] = None,
        layer: Optional[Type[nn.Module]] = None,
        activation: Optional[Type[nn.Module]] = None,
        normalization: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ) -> None: ...

    configure = DeeplayModule.configure
