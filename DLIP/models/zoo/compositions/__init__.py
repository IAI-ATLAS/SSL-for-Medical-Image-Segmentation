"""
    Models to be used must be specified here to be loadable.
"""
from .bolts_ae import BoltsAE
from .s_resnet_unet import SResUnet
from .smp_unet_resnet import SmpUnetResnet
from .moco_v2 import Mocov2
from .dense_cl import DenseCL
from .resnet_classifier import ResnetClassifier
from .custom_resnet import CustomResnet
from .detco_resnet import resnet50_DetCo
from .det_co import DetCo
from .MoDeCo import MoDeCo
from .unet_instance import UnetInstance
from .unet_semantic import UnetSemantic
from .simclr import SimCLR
from .barlow_twins import BarlowTwins
from .byol import BYOL
from .unet_ae import UnetAE