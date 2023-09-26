# %%
import deeplay as dl
import torch

# %%

autoencoder = dl.Autoencoder()
autoencoder._deeplay_forward_hooks

# %%
x = autoencoder.encoder(torch.rand(1, 1, 28, 28))
x = autoencoder.bottleneck(x)
x = autoencoder.decoder(x)
x.shape

# %%
autoencoder

# %%
