from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, _resnet, model_urls
from torchvision._internally_replaced_utils import load_state_dict_from_url

class CustomResnet(ResNet):

    def __init__(
        self,
        num_classes=1000,
        pretrained=False,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            self.load_state_dict(state_dict)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
