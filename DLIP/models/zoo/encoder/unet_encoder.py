from typing import List
from torch.nn.modules.container import ModuleList
import torch.nn as nn
import logging
import numpy as np

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv
from DLIP.models.zoo.building_blocks.down_sample import Down
from DLIP.models.zoo.encoder.basic_encoder import BasicEncoder

class UnetEncoder(BasicEncoder):
    def __init__(
        self,
        input_channels: int,
        encoder_filters: List = [64, 128, 256, 512, 1024],
        dropout: float = 0,
        bilinear: bool = False,
    ):
        super().__init__(input_channels,classification_output=False)
        if bilinear == True:
            logging.info("Bilinear Upsampling is currently not supported. Ignoring.")
        self.bilinear = False
        factor = 2 if self.bilinear else 1
        encoder_filters = [input_channels] + encoder_filters
        factors = (len(encoder_filters)-2)*[1] + [factor]
        dropout_iter = self.get_dropout_iter(dropout, encoder_filters)
        self.backbone.append(DoubleConv(
            encoder_filters[0], 
            encoder_filters[1] // factors[0],
            dropout=next(dropout_iter)
        ))
        for i in range(1,len(encoder_filters)-1):
            self.backbone.append(Down(
                encoder_filters[i], 
                encoder_filters[i+1] // factors[i],
                dropout=next(dropout_iter)
            ))
    

