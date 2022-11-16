import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class NTXentLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        temperature: float = 0.1,
        tau_plus: float = 0.1,
        debiased = True,
        pos_in_denominator = True
    ):
        super(NTXentLoss, self).__init__()
        self.eps = eps
        self.temperature = temperature
        self.debiased = debiased
        self.pos_in_denominator = pos_in_denominator
        self.tau_plus = tau_plus

    def forward(
            self,
            out_1: torch.Tensor,
            out_2: torch.Tensor
        ):
        """
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        # Full similarity matrix
        out = torch.cat([out_1, out_2], dim=0)
        batch_size = out_1.shape[0]
        mask = get_negative_mask(batch_size).cuda()
        cov = torch.mm(out, out.t().contiguous())
        neg  = torch.exp(cov / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
    
        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        if self.debiased:
            N = batch_size * 2 - 2
            Ng = (-self.tau_plus * N * pos + neg.sum(dim = -1)) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
            denominator = Ng + pos
        elif self.pos_in_denominator:
            Ng = neg.sum(dim=-1)
            denominator = Ng + pos
        else:
            Ng = neg.sum(dim=-1)
            denominator = Ng
            
        return -torch.log(pos / (denominator + self.eps)).mean()

class NTXentLossDebiased(NTXentLoss):
    def __init__(self):
        super().__init__(
        debiased=True,
        pos_in_denominator=True)

class NTXentLossBiased(NTXentLoss):
    def __init__(self):
        super().__init__(
        debiased=False,
        pos_in_denominator=False)

class NTXentLossBiasedPosInDenominator(NTXentLoss):
    def __init__(self):
        super().__init__(
        debiased=False,
        pos_in_denominator=True)
