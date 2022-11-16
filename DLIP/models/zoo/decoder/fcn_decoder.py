import torch.nn as nn
from torch.nn import functional as F
import torchvision

class FCNDecoder(nn.Module):

    def __init__(self,
                 img_size,
                 num_classes,
                 norm_cfg = {'type': 'BN', 'requires_grad': True},
                 act_cfg = {'type': 'ReLU'},
                 **kwargs):
        super(FCNDecoder, self).__init__(**kwargs)
        self.fcn_head = torchvision.models.segmentation.fcn_resnet50(False,num_classes=num_classes).classifier
        self.img_size = img_size

    def forward(self, inputs):
        """Forward function."""
        output = self.fcn_head(inputs[0])
        output = F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=False)
        return output