import torchvision
import torchvision.transforms.functional
import functools
import numpy as np
import torch
import kornia


class Transform:
    def __init__(self, forward, inverse=lambda x, **kwargs: x, **kwargs):
        self.forward = forward
        self.inverse = inverse
        self.kwargs = kwargs

    def __call__(self, x):
        n = x.size(0)

        kwargs = self.kwargs.copy()

        for key, value in kwargs.items():
            if callable(value):
                kwargs[key] = torch.tensor([value() for _ in range(n)])

        return self.forward(x, **kwargs), functools.partial(self.inverse, **kwargs)


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        inverses = []
        for transform in self.transforms:
            x, inverse = transform(x)
            inverses.append(inverse)
        return x, self._create_inverse(inverses)

    def _create_inverse(self, inverses):
        def inverse(x):
            for inverse in inverses[::-1]:
                x = inverse(x)
            return x

        return inverse


class RandomTranslation2d(Transform):
    def __init__(
        self,
        dx=lambda: np.random.uniform(-2, 2),
        dy=lambda: np.random.uniform(-2, 2),
        indices=(0, 1),
    ):
        assert len(indices) == 2, "Indices must be a tuple of length 2"
        assert all(isinstance(i, int) for i in indices), "Indices must be integers"
        super().__init__(self._forward, self._backward, dx=dx, dy=dy, indices=indices)

    @staticmethod
    def _forward(x, dx, dy, indices):
        translation = torch.stack([dx, dy], dim=1).type_as(x).to(x.device)
        return kornia.geometry.transform.translate(
            x, translation, align_corners=True, padding_mode="reflection"
        )

    @staticmethod
    def _backward(x: torch.Tensor, dx, dy, indices):
        sub_v = torch.zeros_like(x)
        sub_v[:, indices[0]] = dy
        sub_v[:, indices[1]] = dx
        return x - sub_v


class RandomRotation2d(Transform):
    def __init__(self, angle=lambda: np.random.uniform(-np.pi, np.pi), indices=(0, 1)):
        super().__init__(self._forward, self._backward, angle=angle, indices=indices)

    @staticmethod
    def _forward(x, angle, indices):
        angle = angle.type_as(x).to(x.device)
        return kornia.geometry.transform.rotate(
            x, angle * 180 / np.pi, align_corners=True, padding_mode="reflection"
        )

    @staticmethod
    def _backward(x, angle, indices):
        mat2d = (
            torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        )
        mat2d[:, indices[1], indices[1]] = torch.cos(-angle)
        mat2d[:, indices[1], indices[0]] = -torch.sin(-angle)
        mat2d[:, indices[0], indices[1]] = torch.sin(-angle)
        mat2d[:, indices[0], indices[0]] = torch.cos(-angle)
        out = torch.matmul(x.unsqueeze(1), mat2d).squeeze(1)

        return out
