from typing import Optional, Sequence, Callable, List

from deeplay.components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam

from deeplay import (
    DeeplayModule,
    Layer,
)

import torch
import torch.nn as nn


class VariationalGraphAutoEncoder(Application):
    """ Variational Auto Encoder for Graphs 

    Parameters
    ----------
    encoder : nn.Module
    decoder : nn.Module
    hidden_features : int
        Number of features of the hidden layers
    latent_dim: int
        Number of latent dimensions
    alpha: float
        Weighting for the node feature reconstruction loss. Defaults to 1
    beta: float
        Weighting for the KL loss. Defaults to 1e-7
    gamma: float
        Weighting for the edge feature reconstruction loss. Defaults to 1
    delta: float
        Weighting for the MinCut loss. Defaults to 1
    reconstruction_loss: Reconstruction loss
        Loss metric for the reconstruction of the node and edge features. Defaults to L1 (Mean absolute error).
    optimizer: Optimizer
        Optimizer to use for training.
    """
    
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    hidden_features: int
    latent_dim: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    reconstruction_loss: torch.nn.Module
    optimizer: Optimizer

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        hidden_features: Optional[int] = 96,
        latent_dim=int,
        alpha: Optional[float] = 1,
        beta: Optional[float] = 1e-7,
        gamma: Optional[float] = 1,
        delta: Optional[float] = 1,
        reconstruction_loss: Optional[Callable] = nn.L1Loss(),
        optimizer=None,
        **kwargs,
    ):
        self.encoder = encoder
        
        self.fc_mu = Layer(nn.Linear, hidden_features, latent_dim)
        self.fc_mu.set_input_map('x')
        self.fc_mu.set_output_map('mu')
        
        self.fc_var = Layer(nn.Linear, hidden_features, latent_dim)
        self.fc_var.set_input_map('x')
        self.fc_var.set_output_map('log_var')

        self.fc_dec = Layer(nn.Linear, latent_dim, hidden_features)
        self.fc_dec.set_input_map('z')
        self.fc_dec.set_output_map('x')

        self.decoder = decoder

        self.reconstruction_loss = reconstruction_loss or nn.L1Loss()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta                                               
        self.gamma = gamma
        self.delta = delta

        super().__init__(**kwargs)

        class Reparameterize(DeeplayModule):
            def forward(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return eps * std + mu
        
        self.reparameterize = Reparameterize()
        self.reparameterize.set_input_map('mu', 'log_var')
        self.reparameterize.set_output_map('z')

        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.parameters()
    

    def encode(self, x):
        x = self.encoder(x)
        x = self.fc_mu(x)
        x = self.fc_var(x)
        return x

    def decode(self, x):
        x = self.fc_dec(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch) 
        node_features, edge_features = y
        x = self(x)
        node_features_hat = x['x']
        edge_features_hat = x['edge_attr']
        mu = x['mu']
        log_var = x['log_var']
        mincut_cut_loss = sum(value for key, value in x.items() if key.startswith('L_cut'))
        mincut_ortho_loss = sum(value for key, value in x.items() if key.startswith('L_ortho'))
        rec_loss_nodes, rec_loss_edges, KLD = self.compute_loss(node_features_hat, node_features, edge_features_hat, edge_features, mu, log_var)

        tot_loss = self.alpha * rec_loss_nodes + self.gamma * rec_loss_edges + self.beta * KLD + self.delta * (mincut_cut_loss + mincut_ortho_loss)

        loss = {"rec_loss_nodes": rec_loss_nodes, "rec_loss_edges": rec_loss_edges, "KL": KLD,
                "MinCut cut loss": mincut_cut_loss, "MinCut orthogonality loss": mincut_ortho_loss, 
                "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss

    def compute_loss(self, n_hat, n, e_hat, e, mu, log_var):
        
        rec_loss_nodes = self.reconstruction_loss(n_hat, n)
        rec_loss_edges = self.reconstruction_loss(e_hat, e)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return rec_loss_nodes, rec_loss_edges, KLD

    def forward(self, x):
        x = self.encode(x)
        x = self.reparameterize(x)
        x = self.decode(x)
        return x
