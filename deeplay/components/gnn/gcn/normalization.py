import torch
import torch.nn as nn

from deeplay import DeeplayModule


class sparse_laplacian_normalization(DeeplayModule):
    def add_self_loops(self, A, num_nodes):
        """
        Add self-loops to the adjacency matrix of a graph.
        """
        loop_index = torch.arange(num_nodes, device=A.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        A = torch.cat([A, loop_index], dim=1)

        return A

    def degree(self, A, num_nodes):
        """
        Compute the degree of each node in a graph given its edge index.
        """
        A = torch.unique(A, dim=1)
        row, col = A

        deg = torch.zeros(num_nodes, dtype=torch.long, device=A.device)

        # Count occurrences of each unique edge to compute degree
        deg.index_add_(0, row, torch.ones_like(row))
        return A, deg

    def normalize(self, x, A):
        if A.size(0) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects 'A' to be an edge index matrix of size 2 x N. "
                "Please ensure that 'A' follows this format for proper functioning. "
                "For dense adjacency matrices, consider using dense_laplacian_normalization instead,",
                " i.e., GNN.normalize.configure(deeplay.dense_laplacian_normalization)",
            )

        A, deg = self.degree(A, x.size(0))
        row, col = A

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return A, norm

    def forward(self, x, A):
        A = self.add_self_loops(A, x.size(0))
        # computes: D^-1/2 A' D^-1/2
        A, norm = self.normalize(x, A)
        # sparse matrix multiplication
        laplacian = torch.sparse_coo_tensor(
            A,
            norm,
            (x.size(0),) * 2,
            device=A.device,
        )
        return laplacian


class dense_laplacian_normalization(DeeplayModule):
    def degree(self, A):
        """
        Compute the degree of each node in a graph given its adjacency matrix.
        """
        deg = torch.sum(A, dim=1)
        return deg

    def normalize(self, x, A):
        if A.size(0) != A.size(1):
            raise ValueError(
                f"{self.__class__.__name__} expects 'A' to be a square adjacency matrix. "
                "Please ensure that 'A' follows this format for proper functioning. "
                "For edge index matrices, consider using sparse_laplacian_normalization instead,",
                " i.e., GNN.normalize.configure(deeplay.sparse_laplacian_normalization)",
            )

        deg = self.degree(A)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

        return norm

    def forward(self, x, A):
        A = A + torch.eye(x.size(0), device=A.device)
        # computes: I - D^-1/2 A D^-1/2
        laplacian = self.normalize(x, A) * A

        return laplacian
