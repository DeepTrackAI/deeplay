from ..applications import DeeplayLightningModule
from ..classification import ImageClassifier
from ...config import Config
from ...templates import Layer
from ...components import VectorToImageDecoder, ImageToVectorEncoder, ImageGeneratorHead
import torch
import torch.nn as nn
from torch.optim import Adam

class VanillaGAN(DeeplayLightningModule):

    @staticmethod
    def defaults():
        return (
            Config()
            .hidden_dim(128)
            .generator(Layer("backbone") >> Layer("head"))
            .generator.backbone(VectorToImageDecoder)
            .generator.backbone.base_size((7, 7))
            .generator.head(ImageGeneratorHead)
            .discriminator(ImageClassifier)
            .discriminator.num_classes(2)
            .sampler(torch.distributions.Normal, loc=0, scale=1)
            .on_first_forward("generator.backbone.output_size", lambda _, x: x.shape[2:])

            .discriminator_loss(nn.CrossEntropyLoss)
            .generator_loss(nn.CrossEntropyLoss)
            .discriminator_optimizer(Adam, lr=1e-3)
            .generator_optimizer(Adam, lr=1e-3)
        )
    
    
    def __init__(self, hidden_dim=128, generator=None, discriminator=None, sampler=None, discriminator_loss=None, generator_loss=None, discriminator_optimizer=None, generator_optimizer=None):
        DeeplayLightningModule.__init__(self, hidden_dim=hidden_dim, generator=generator, discriminator=discriminator, sampler=sampler, discriminator_loss=discriminator_loss, generator_loss=generator_loss, discriminator_optimizer=discriminator_optimizer, generator_optimizer=generator_optimizer)
        
        self.hidden_dim = self.attr("hidden_dim")
        
        self.generator = self.new("generator")
        self.discriminator = self.new("discriminator")
        self.sampler = self.new("sampler")

        self.discriminator_loss = self.new("discriminator_loss")
        self.generator_loss = self.new("generator_loss")


    def configure_optimizers(self):
        return (
            self.new("generator_optimizer", extra_kwargs={"params": self.generator.parameters()}),
            self.new("discriminator_optimizer", extra_kwargs={"params": self.discriminator.parameters()}),
        )
        
    def forward(self, x=None):
        if x is None:
            x = self.sampler.sample((1, self.hidden_dim))
            x = x.to(self.device).float()
        return self.generator(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self._get_image(batch)
        z = self.sampler.sample((x.shape[0], self.hidden_dim))

        if optimizer_idx == 0:
            # store generated images for the discriminator optimizer
            self._generated_imgs = self(z)

            valid = torch.ones(x.size(0), 1).type_as(x)

            g_loss = self.generator_loss(self.discriminator(self._generated_imgs), valid)
            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)
        
            return g_loss
        
        if optimizer_idx == 1:
            valid = torch.ones(x.size(0), 1).type_as(x)
            fake = torch.zeros(x.size(0), 1).type_as(x)

            real_loss = self.discriminator_loss(self.discriminator(x), valid)
            fake_loss = self.discriminator_loss(self.discriminator(self._generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
            return d_loss

    
    def _get_image(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch