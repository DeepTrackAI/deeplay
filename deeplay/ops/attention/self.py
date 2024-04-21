from .cross import MultiheadCrossAttention


class MultiheadSelfAttention(MultiheadCrossAttention):
    def __init__(
        self,
        features: int,
        num_heads: int,
        return_attn: bool = False,
        batch_first: bool = False,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__(
            features,
            num_heads,
            return_attn=return_attn,
            batch_first=batch_first,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(self, x, batch_index=None):
        return super().forward(x, x, x, batch_index=batch_index)
