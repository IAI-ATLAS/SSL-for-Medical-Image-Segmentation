import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BYOLLoss(nn.Module):
    def __init__(
        self
    ) -> None:
        super(BYOLLoss, self).__init__()

    def forward(
            self,
            out_1: torch.Tensor,
            out_2: torch.Tensor
        ):
        return -2 * F.cosine_similarity(out_1, out_2).mean()
