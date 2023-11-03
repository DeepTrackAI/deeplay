# %%

import deeplay as dl
import torch
import torch.nn as nn


class Module(dl.DeeplayModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x = self.attr("x")
        self.net = nn.Linear(10, 1)

    def forward(self, x):
        return self.net(x)


class App(dl.Application):
    def __init__(self, **kwargs):
        super().__init__()
        self.x = self.attr("x")
        self.net = self.new("net")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        self.log("train_loss", loss)
        self.log("x", self.x, on_epoch=True, on_step=True, prog_bar=True)

        return loss


# %%
train_data = torch.rand(1000, 10)
train_targets = torch.rand(1000, 1)
dataset = torch.utils.data.TensorDataset(train_data, train_targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
trainer = dl.Trainer(max_epochs=10, accelerator="cpu")
# %%
# config = dl.Config().x(

# )

model = App.from_config(
    dl.Config()
    .x(dl.schedulers.LinearScheduler(0, 1, 1000))
    .net(Module)
    .net.x(dl.schedulers.LinearScheduler(0, 1, 1000))
    .optimizer(torch.optim.Adam, lr=1e-3)
)

# %%

model.net.x

# %%
trainer.fit(model, dataloader)

# %%

import lightning as L

L.LightningModule
