import torch
import torch.nn as nn
from packaging import version

class DenseCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv

        Snatched from: https://github.com/WXinlong/DenseCL/blob/1706b5be1db843d398142a64c9857551d3a74c03/openselfsup/models/necks.py#L357

    '''
    def __init__(
        self,
        dim_mlp,
        emb_dim,
        num_grid=None
    ):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, emb_dim)
        )

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_mlp, dim_mlp, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim_mlp, emb_dim, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # s = filter dim
        # b = batch size
        # d = hidden dim size
        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))
        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]