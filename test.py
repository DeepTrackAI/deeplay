# %%

import torch

conv = torch.nn.Conv2d(3, 16, 3, padding=1)

conv.in_channels = 2

# %%
tensor = torch.randn(1, 2, 32, 32)
conv(tensor).shape
