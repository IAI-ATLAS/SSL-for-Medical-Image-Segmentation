
from torch import nn


class Mocov2Neck(nn.Module):

    def __init__(
        self,
        emb_dim,
        dim_mlp,
        with_avg_pool=False
        ):
        super().__init__()
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.with_avg_pool = with_avg_pool
        self.mlp = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, emb_dim)
        )

    def forward(self,x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x)