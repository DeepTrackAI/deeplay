"""MIRO: Multimodal Integration through Relational Optimization

This module provides the MIRO framework for point cloud clustering, leveraging
advanced geometric deep learning techniques. MIRO transforms complex point
clouds into optimized representations, enabling more effective clustering
using traditional algorithms.

Based on the original MIRO paper by Pineda et al. [1], this implementation offers
easy-to-use methods for training the MIRO model and performing geometric-aware
clustering. It integrates recurrent graph neural networks to refine point
cloud data and enhance clustering accuracy.

[1] Pineda, Jesús, et al. "Spatial Clustering of Molecular Localizations with
    Graph Neural Networks." arXiv preprint arXiv:2412.00173 (2024).
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from typing import Callable, Optional, List

from deeplay.applications import Application
from deeplay.external import Adam, Optimizer
from deeplay.models import RecurrentMessagePassingModel


class MIRO(Application):
    """Point cloud clustering using MIRO (Multimodal Integration through
    Relational Optimization).

    MIRO is a geometric deep learning framework that enhances clustering
    algorithms by transforming complex point clouds into an optimized structure
    amenable to conventional clustering methods. MIRO employs recurrent graph
    neural networks (rGNNs) to learn a transformation that squeezes localization
    belonging to the same cluster toward a common center, resulting in a compact
    representation of clusters within the point cloud.

    Parameters
    ----------
    num_outputs : int
        Dimensionality of the output features, representing a displacement
        vector in Cartesian space for each node. This vector points toward
        the center of each cluster.
    connectivity_radius : float
        Maximum distance between two nodes to consider them connected in the
        graph.
    model : nn.Module
        A model implementing the forward method. It should return a list of
        tensors of shape `(num_nodes, num_outputs)` representing the predicted
        displacement vectors for each node at each recurrent iteration. If not
        specified, a default model resembling the one from the original MIRO
        paper is used.
    nd_loss_weight : float
        Weight for the auxiliary loss that enforces preservation of pairwise
        distances between connected nodes. Default is 10.
    loss : torch.nn.Module
        Loss function for training. Default is `torch.nn.L1Loss`.
    optimizer : Optimizer
        Optimizer for training. Default is Adam with a learning rate of 1e-4.

    Returns
    -------
    forward : method
        Computes and returns the predicted displacement vectors for each node
        in the input graph. The output is a list of tensors representing the
        displacement vectors at each recurrent iteration.

    squeeze : method
        Applies the predicted displacement vectors from the last recurrent
        iteration (by default) to the input point cloud. This operation
        optimizes the point cloud for clustering by aligning nodes closer to
        their respective cluster centers.

    clustering : method
        Groups nodes into clusters using the DBSCAN algorithm, based on the
        predicted displacement vectors. Each node is assigned a cluster label,
        where -1 indicates background noise. Returns an array of cluster labels
        for the nodes.

    Example
    -------
    >>> # Predicts displacement vectors for each node in a point cloud at each
    >>> # recurrent iteration
    >>> displacement_vectors = model(test_graph)
    >>> print(type(displacement_vectors))
    <class 'list'>

    >>> # Applies the predicted displacement vectors to the input point cloud
    >>> squeezed = model.squeeze(test_graph)
    >>> print(squeezed.shape)
    (num_nodes, 2)

    >>> # Performs clustering using DBSCAN after MIRO squeezing
    >>> eps = 0.3  # Maximum distance for cluster connection
    >>> min_samples = 5  # Minimum points in a neighborhood for core points
    >>> clusters = model.clustering(test_graph, eps, min_samples)

    >>> # Output cluster labels
    >>> print(clusters)
    array([ 0,  0,  1,  1,  1, -1,  2,  2,  2, ...])
    # Nodes in cluster 0, 1, 2, etc.; -1 are outliers
    """

    num_outputs: int
    connectivity_radius: float
    model: nn.Module
    nd_loss_weight: float
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        num_outputs: int = 2,
        connectivity_radius: float = 1.0,
        model: Optional[nn.Module] = None,
        nd_loss_weight: float = 10,
        loss: torch.nn.Module = torch.nn.L1Loss(),
        optimizer=None,
        **kwargs,
    ):

        self.num_outputs = num_outputs
        self.connectivity_radius = connectivity_radius
        self.model = model or self._get_default_model()
        self.nd_loss_weight = nd_loss_weight

        super().__init__(loss=loss, optimizer=optimizer or Adam(lr=1e-4), **kwargs)

    def _get_default_model(self):
        rgnn = RecurrentMessagePassingModel(
            hidden_features=256, out_features=self.num_outputs, num_iter=20
        )
        return rgnn

    def forward(self, x: Data) -> List[torch.Tensor]:
        """Forward pass to compute predicted displacement vectors for each node.

        Parameters
        ----------
         x : torch_geometric.data.Data
            Input graph data. It is expected to have the attributes:
            `x` (node features), `edge_index` (graph connectivity),
            `edge_attr` (edge features), and `positions` (node spatial coordinates).

        Returns
        -------
        list[torch.Tensor]
            Predicted displacement vectors at each recurrent iteration.
        """
        return self.model(x)

    def squeeze(
        self,
        x: Data,
        from_iter: int = -1,
        scaling: np.ndarray = np.array([1.0, 1.0]),
    ) -> np.ndarray:
        """Computes and applies the predicted displacement vectors to the
        input point cloud.

        Parameters
        ----------
         x : torch_geometric.data.Data
            Input graph data. It is expected to have the attributes:
            `x` (node features), `edge_index` (graph connectivity),
            `edge_attr` (edge features), and `positions` (node spatial coordinates).
        from_iter : int, optional
            Index of the recurrent iteration to be used as displacement vectors.
            Default is -1 (last iteration).
        scaling : np.ndarray, optional
            Scaling factors for each dimension. Default is [1.0, 1.0].

        Returns
        -------
        np.ndarray
            Squeezed point cloud with optimized cluster alignment.
        """
        predicted_displacements = self(x)[from_iter].detach().cpu().numpy()
        positions = x.position.cpu().numpy()
        squeezed_positions = (
            positions - predicted_displacements * self.connectivity_radius
        )
        return squeezed_positions * scaling

    def clustering(
        self,
        x: Data,
        eps: float,
        min_samples: int,
        from_iter: int = -1,
        **kwargs,
    ) -> np.ndarray:
        """Perform clustering using DBSCAN after applying MIRO squeezing.

        Parameters
        ----------
        x : torch_geometric.data.Data
            Input graph data.
        eps : float
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data set
            and distance function.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
        from_iter : int, optional
            Index of the recurrent iteration to be used as displacement vectors.
            Default is -1 (last iteration).

        Returns
        -------
        np.ndarray
            Cluster labels for each node. -1 indicates outliers.
        """
        squeezed = self.squeeze(x, from_iter, **kwargs)
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(squeezed)
        return clusters.labels_

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Defines the training step for a single batch."""
        x, y = self.train_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, x.edge_index, x.position)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.log_metrics(
            "train", y_hat, y, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def compute_loss(
        self,
        y_hat: List[torch.Tensor],
        y: torch.Tensor,
        edges: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total loss for the model."""
        loss = 0
        for pred in y_hat:
            loss += self.loss(pred, y) + self.nd_loss_weight * self.compute_nd_loss(
                pred, y, edges, position
            )
        return loss / len(y_hat)

    def compute_nd_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        edges: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Computes auxiliary loss for pairwise distance preservation."""
        squeezed_gt = position - y * self.connectivity_radius
        squeezed_gt_distances = torch.norm(
            squeezed_gt[edges[0]] - squeezed_gt[edges[1]], dim=1
        )
        squeezed_pred = position - y_hat * self.connectivity_radius
        squeezed_pred_distances = torch.norm(
            squeezed_pred[edges[0]] - squeezed_pred[edges[1]], dim=1
        )
        return self.loss(squeezed_pred_distances, squeezed_gt_distances)
