from ....core import DeeplayModule
from ....config import Config
import torch
import torch.nn as nn
import numpy as np
import warnings

import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=1)
    return emb


class _BasePositionalEncodingSinusoidal(DeeplayModule):
    @staticmethod
    def defaults():
        return Config().num_encodings(32).period(10000 * 2 * np.pi)

    def __init__(self, **kwargs):
        self.num_encodings = self.attr("num_encodings")
        self.period = self.attr("period")
        super(_BasePositionalEncodingSinusoidal, self).__init__(**kwargs)

    def forward(self, x):
        """
        :param tensor: A tensor of size (batch_size, ch, *), where * denotes any number of additional dimensions
        :return: Positional Encoding Matrix of size (batch_size, ch + out_features, *)
        """
        encoding = self.encoding(x)
        return torch.cat((x, encoding), dim=1)

    def encoding(self, tensor):
        """
        :param tensor: A tensor of size (batch_size, ch, *), where * denotes any number of additional dimensions
        :return: Positional Encoding Matrix of size (batch_size, ch, *)
        """
        raise NotImplementedError


class PositionalEncodingSinusoidal1d(_BasePositionalEncodingSinusoidal):
    def __init__(self, **kwargs):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSinusoidal1d, self).__init__()
        num_encodings = self.num_encodings
        self.org_channels = num_encodings
        num_encodings = int(np.ceil(num_encodings / 2) * 2)
        self.channels = num_encodings
        inv_freq = 1.0 / (
            (self.period / (2 * np.pi))
            ** (torch.linspace(1, 0, num_encodings // 2).float())
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def encoding(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, ch, x)
        :return: Positional Encoding Matrix of size (batch_size, num_encodings, x)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, _, x = tensor.shape

        pos_x = torch.arange(x, device=tensor.device).type_as(self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        emb = get_emb(sin_inp_x)

        self.cached_penc = emb[None].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingSinusoidal2d(_BasePositionalEncodingSinusoidal):
    def __init__(self, num_encodings):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSinusoidal2d, self).__init__()
        num_encodings = self.num_encodings
        self.org_channels = num_encodings
        num_encodings = int(np.ceil(num_encodings / 4) * 2)
        self.channels = num_encodings
        inv_freq = 1.0 / (
            (self.period / (2 * np.pi))
            ** (torch.linspace(1, 0, num_encodings // 2).float())
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def encoding(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, num_encodings, x, y)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        (
            batch_size,
            _,
            x,
            y,
        ) = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type_as(self.inv_freq)
        pos_y = torch.arange(y, device=tensor.device).type_as(self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        emb_x = get_emb(sin_inp_x).view(1, self.channels, x, 1).repeat(1, 1, 1, y)
        emb_y = get_emb(sin_inp_y).view(1, self.channels, 1, y).repeat(1, 1, x, 1)
        emb = torch.cat((emb_x, emb_y), dim=1)

        self.cached_penc = emb.repeat(batch_size, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingSinusoidal3d(_BasePositionalEncodingSinusoidal):
    def __init__(self, num_encodings):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSinusoidal3d, self).__init__()
        num_encodings = self.num_encodings
        self.org_channels = num_encodings
        num_encodings = int(np.ceil(num_encodings / 6) * 2)
        if num_encodings % 2:
            num_encodings += 1
        self.channels = num_encodings
        inv_freq = 1.0 / (
            (self.period / (2 * np.pi))
            ** (torch.linspace(1, 0, num_encodings // 2).float())
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def encoding(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type_as(self.inv_freq)
        pos_y = torch.arange(y, device=tensor.device).type_as(self.inv_freq)
        pos_z = torch.arange(z, device=tensor.device).type_as(self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        sin_inp_z = torch.einsum("i,j->ij", self.inv_freq, pos_z)
        emb_x = get_emb(sin_inp_x).view(1, self.channels, x, 1, 1).repeat(1, 1, 1, y, z)
        emb_y = get_emb(sin_inp_y).view(1, self.channels, 1, y, 1).repeat(1, 1, x, 1, z)
        emb_z = get_emb(sin_inp_z).view(1, self.channels, 1, 1, z).repeat(1, 1, x, y, 1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=1)

        self.cached_penc = emb.repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc
