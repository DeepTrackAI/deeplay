from ....core import DeeplayModule
import torch
import torch.nn as nn


class _PositionalEncodingLinear(DeeplayModule):
    def encoding(self, x, positions=None):
        """Encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tensor.
        """

        num_spatial_dims = len(x.shape) - 2
        if positions is None:
            positions = self.get_pixel_positions(x)

        scaling = x.shape[2:]
        scaling = torch.tensor(scaling, device=positions.device, dtype=positions.dtype)
        scaling = scaling.view(1, scaling.numel(), *((1,) * num_spatial_dims))
        encoding = positions * 2 / scaling - 1

        return encoding

    def forward(self, x, positions=None):
        encoding = self.encoding(x, positions=positions)
        x = torch.cat([x, encoding], dim=1)
        return x

    def get_pixel_positions(self, x):
        """Get pixel positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Pixel positions.
        """
        raise NotImplementedError


class PositionalEncodingLinear1d(DeeplayModule):
    def __init__(self):
        """Linear Positional Encoding for 1d sequences

        __
        """
        super().__init__()

    def forward(self, x, positions=None):
        if len(x.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        return super().forward(x, positions=positions)

    def get_pixel_positions(self, x):
        """Get pixel positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Pixel positions.
        """
        batch_size, _, h = x.shape
        grid = torch.arange(h, device=x.device, dtype=x.dtype)
        grid = grid.view(1, 1, h).repeat(batch_size, 1, 1)

        return grid


class PositionalEncodingLinear2d(_PositionalEncodingLinear):
    def __init__(self):
        """Linear Positional Encoding for 2d images

        __
        """
        super().__init__()

    def forward(self, x, positions=None):
        if len(x.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        return super().forward(x, positions=positions)

    def get_pixel_positions(self, x):
        """Get pixel positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Pixel positions.
        """
        batch_size, _, h, w = x.shape
        grid = (
            torch.stack(
                torch.meshgrid(torch.arange(h), torch.arange(w)),
                dim=0,
            )
            .float()
            .to(x.device)
        )
        grid = grid.view(1, 2, h, w).repeat(batch_size, 1, 1, 1)

        return grid


class PositionalEncodingLinear3d(_PositionalEncodingLinear):
    def __init__(self):
        """Linear Positional Encoding for 3d volumes

        __
        """
        super().__init__()

    def forward(self, x, positions=None):
        if len(x.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        return super().forward(x, positions=positions)

    def get_pixel_positions(self, x):
        """Get pixel positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Pixel positions.
        """
        batch_size, _, h, w, d = x.shape
        grid = torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), torch.arange(d)),
            dim=0,
        ).float()
        grid = grid.view(1, 3, h, w, d).repeat(batch_size, 1, 1, 1, 1)

        return grid
