import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CrossCorrelationLoss(nn.Module):
    def __init__(
        self,
        lambd: float = 0.0051
    ) -> None:
        super(CrossCorrelationLoss, self).__init__()
        self.lambd = lambd

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(
            self,
            out_1: torch.Tensor,
            out_2: torch.Tensor
        ):
        """
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]

        Assuming both out_1 and out_2 are batch normalized.

        """
        # Norm
        z1 = (out_1 - out_1.mean(0)) / out_1.std(0) # NxD
        z2 = (out_2 - out_2.mean(0)) / out_2.std(0) # NxD
        
        N = z1.size(0)

        # cross-correlation matrix
        c = torch.mm(z1.T, z2).div(N)  # DxD
        # First term of the loss: \sum_i (1-C_{ii})^2
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # Second term of the loss: \sum_i \sum_{j \neq i} C_{ij}^2
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag
