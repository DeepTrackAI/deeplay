from ..application import Application
from ...components import ConvolutionalNeuralNetwork

import torch
import torch.nn as nn

import numpy as np


from skimage import morphology
import scipy.ndimage
import scipy


class LodeSTAR(Application):
    def __init__(
        self,
        model=None,
        transforms=None,
        between_loss=None,
        within_loss=None,
        between_loss_weight=1,
        within_loss_weight=0.01,
        **kwargs
    ):
        try:
            import deeptrack as dt
        except ImportError as e:
            raise ImportError(
                "LodeSTAR requires deeptrack to be installed. "
                "Please install deeptrack by running `pip install deeptrack`."
            ) from e

        if transforms is None:
            transforms = [
                dt.Affine(rotation_range=360, translation_range=0.1, zoom_range=0.1),
            ]

        self.num_outputs = num_outputs
        self.model = model or self._build_default_model()
        self.between_loss = between_loss or nn.L1Loss(reduction="mean")
        self.within_loss = within_loss or nn.L1Loss(reduction="mean")
        self.between_loss_weight = between_loss_weight
        self.within_loss_weight = within_loss_weight

        super().__init__(**kwargs)

    def _build_default_model(self):
        cnn = ConvolutionalNeuralNetwork(
            None,
            [32, 32, 64, 64, 64, 64, 64, 64, 64],
            self.num_outputs,
        )
        cnn.blocks[2].pool.configure(nn.MaxPool2d, kernel_size=2)

        return cnn

    def forward(self, x):
        _, _, Hx, Wx = x.shape

        y = self.model(x)

        _, _, Hy, Wy = y.shape

        x_range = torch.arange(Hy, device=x.device) * Hx / Hy
        y_range = torch.arange(Wy, device=x.device) * Wx / Wy

        Y, X = torch.meshgrid(y_range, x_range)

        delta_x = y[:, 0]
        delta_y = y[:, 1]
        weights = y[:, -1].sigmoid()

        X = X + delta_x
        Y = Y + delta_y

        return torch.cat([X[:, None], Y[:, None], y[:, 2:-1], weights[:, None]], dim=1)

    def normalize(self, weights):
        weights = weights + 1e-6
        return weights / weights.sum(dim=(2, 3), keepdim=True)

    def reduce(self, X, weights):
        return (X * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

    def compute_loss(self, y_hat, y):
        y_pred, weights = y_hat[:, :-1], y_hat[:, -1]

        weights = self.normalize(weights)
        y_reduced = self.reduce(y_pred, weights)

        consistency = (y_pred - y_reduced[..., None, None]) * weights
        consistency_loss = self.within_loss(consistency, torch.zeros_like(consistency))

        y_reduced_on_initial = self.apply_inverse(y_reduced, y)

        average_on_initial = y_reduced_on_initial.mean(dim=0, keepdim=True)

        inter_consistency_loss = self.between_loss(
            y_reduced_on_initial, average_on_initial
        )

        return {
            "between_image_disagreement": consistency_loss * self.within_loss_weight,
            "within_image_disagreement": inter_consistency_loss
            * self.between_loss_weight,
        }

    def apply_inverse(self, y_reduced, y):
        for f in reversed(y):
            y_reduced = f(y_reduced)
        return y_reduced

    def detect(self, x, alpha=0.5, beta=0.5, cutoff=0.97, mode="quantile"):
        y = self.model(x.to(self.device))
        y_pred, weights = y[:, :-1], y[:, -1]
        detections = [
            self.detect_single(y_pred[i], weights[i], alpha, beta, cutoff, mode)
            for i in range(len(y_pred))
        ]

        return detections

    def pooled(self, x, mask=1):
        y = self.model(x.to(self.device))
        y_pred, weights = y[:, :-1], y[:, -1]
        masked_weights = weights * mask

        pooled = self.reduce(y_pred, self.normalize(masked_weights))

        return pooled

    def detect_single(
        self, y_pred, weights, alpha=0.5, beta=0.5, cutoff=0.97, mode="quantile"
    ):
        score = self.get_detection_score(y_pred, weights, alpha, beta)
        return self.find_local_maxima(y_pred, score, cutoff, mode)

    @staticmethod
    def find_local_maxima(pred, score, cutoff=0.9, mode="quantile"):
        """Finds the local maxima in a score-map, indicating detections

        Parameters
            ----------
        pred, score: array-like
            Output from model, score-map
        cutoff, mode: float, string
            Treshholding parameters. Mode can be either "quantile" or "ratio" or "constant". If "quantile", then
            `ratio` defines the quantile of scores to accept. If "ratio", then cutoff defines the ratio of the max
            score as threshhold. If constant, the cutoff is used directly as treshhold.

        """
        score = score[3:-3, 3:-3]
        th = cutoff
        if mode == "quantile":
            th = np.quantile(score, cutoff)
        elif mode == "ratio":
            th = np.max(score.flatten()) * cutoff
        hmax = morphology.h_maxima(np.squeeze(score), th) == 1
        hmax = np.pad(hmax, ((3, 3), (3, 3)))
        detections = pred[hmax, :]
        return np.array(detections)

    @staticmethod
    def local_consistency(pred):
        """Calculate the consistency metric

        Parameters
        ----------
        pred : array-like
            first output from model
        """
        pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
        kernel = np.ones((1, 3, 3)) / 3**2
        pred_local_squared = scipy.signal.convolve(pred, kernel, "same") ** 2
        squared_pred_local = scipy.signal.convolve(pred**2, kernel, "same")
        squared_diff = (squared_pred_local - pred_local_squared).sum(-1)
        np.clip(squared_diff, 0, np.inf, squared_diff)
        return 1 / (1e-6 + squared_diff)

    @classmethod
    def get_detection_score(cls, pred, weights, alpha=0.5, beta=0.5):
        """Calculates the detection score as weights^alpha * consistency ^ beta.

        Parameters
        ----------
        pred, weights: array-like
            Output from model
        alpha, beta: float
            Geometric weight of the weight-map vs the consistenct metric for detection.
        """
        return weights[..., 0] ** alpha * cls.local_consistency(pred) ** beta

    def train_preprocess(self, batch):
        x, *_ = batch

        return super().train_preprocess(batch)
