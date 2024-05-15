from typing import Sequence

from deeplay import DeeplayModule

import torch
import torch.nn as nn


class LearnableDistancewWeighting(DeeplayModule):
    def __init__(
        self,
        init_sigma: float = 0.12,
        init_beta: float = 4.0,
        sigma_limit: Sequence[float] = [0.002, 1],
        beta_limit: Sequence[float] = [1, 10],
    ):
        super().__init__()

        self.init_sigma = init_sigma
        self.init_beta = init_beta
        self.sigma_limit = sigma_limit
        self.beta_limit = beta_limit

        self.sigma = nn.Parameter(torch.tensor(init_sigma), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(init_beta), requires_grad=True)

    def forward(self, inputs):

        sigma = torch.clamp(self.sigma, *self.sigma_limit)
        beta = torch.clamp(self.beta, *self.beta_limit)

        return torch.exp(
            -1
            * torch.pow(
                torch.square(inputs) / (2 * torch.square(sigma)),
                beta,
            )
        )

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"init_sigma={self.init_sigma}",
                f"init_beta={self.init_beta}",
            ]
        )
