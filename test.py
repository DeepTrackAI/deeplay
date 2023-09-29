# %%

import deeplay as dl

encoder = dl.encoders.Base2dConvolutionalEncoder()
encoder

# %%

import torch.nn.modules.linear

torch.nn.modules.linear.Identity(torch.randn(1, 2, 3))

# %%
encoder.new("output_block", now=True)
