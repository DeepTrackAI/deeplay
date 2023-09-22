# %%
import torch

x = torch.randn(2, 50)
x.unflatten(1, (-1, 5, 5)).size()

# %%
