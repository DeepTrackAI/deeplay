import torch
import torch.nn as nn


class sparse_normalization(nn.Module):
    def degree(self, A, num_nodes):
        """
        Compute the degree of each node in a graph given its edge index.
        """
        (row, col), inverse_indices = torch.unique(A, dim=1, return_inverse=True)
        deg = torch.zeros(num_nodes, dtype=torch.long, device=A.device)

        # Count occurrences of each unique edge to compute degree
        deg.index_add_(0, row, torch.ones_like(inverse_indices))
        return deg

    def normalize(self, x, A):
        if A.size(0) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects 'A' to be an edge index matrix of size 2 x N. "
                "Please ensure that 'A' follows this format for proper functioning. "
                "For dense adjacency matrices, consider using dense_normalization instead,",
                " i.e., GNN.normalize.configure(deeplay.dense_normalization)",
            )

        row, col = A
        deg = self.degree(A, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm

    def forward(self, x, A):
        norm = self.normalize(x, A)
        # sparse matrix multiplication
        norm = torch.sparse_coo_tensor(
            A,
            norm,
            (x.size(0),) * 2,
            device=A.device,
        )
        # computes: I - D^-1/2 A D^-1/2
        laplacian = torch.eye(x.size(0), device=norm.device).to_sparse() - norm

        return laplacian


class dense_normalization(nn.Module):
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
                "For edge index matrices, consider using sparse_normalization instead,",
                " i.e., GNN.normalize.configure(deeplay.sparse_normalization)",
            )

        deg = self.degree(A)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

        return norm

    def forward(self, x, A):
        norm = self.normalize(x, A)
        # computes: I - D^-1/2 A D^-1/2
        laplacian = torch.eye(x.size(0), device=norm.device) - norm * A

        return laplacian
