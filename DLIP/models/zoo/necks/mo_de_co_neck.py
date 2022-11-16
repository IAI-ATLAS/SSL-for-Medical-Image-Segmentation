import torch
import torch.nn as nn
from packaging import version

class MoDeCoNeck(nn.Module):
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
        super(MoDeCoNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_mlp = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, emb_dim)
        )
        
        self.mlp1 = nn.Sequential(
            nn.Conv2d(2048, 2048, 1), nn.ReLU(inplace=True),
            nn.Conv2d(2048, emb_dim, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, emb_dim, 1)
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp3 = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, emb_dim, 1)
        )
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp4 = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, emb_dim, 1)
        )
        self.avgpool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp5 = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, emb_dim, 1)
        )
        self.avgpool5 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # s = filter dim
        # b = batch size
        # d = hidden dim size
        avgpooled_global = self.avgpool(x[0])
        global_x = self.global_mlp(avgpooled_global.squeeze())
        
        x_1 = self.mlp1(x[0])
        avgpooled_x1 = self.avgpool1(x_1).squeeze()
        
        x_2 = self.mlp2(x[1][0])
        avgpooled_x2 = self.avgpool1(x_2).squeeze()
        
        x_3 = self.mlp3(x[1][1])
        avgpooled_x3 = self.avgpool1(x_3).squeeze()
        
        x_4 = self.mlp4(x[1][2])
        avgpooled_x4 = self.avgpool1(x_4).squeeze()
        
        x_5 = self.mlp5(x[1][3])
        avgpooled_x5 = self.avgpool1(x_5).squeeze()
        
        return global_x, [avgpooled_x1,avgpooled_x2,avgpooled_x3,avgpooled_x4,avgpooled_x5]