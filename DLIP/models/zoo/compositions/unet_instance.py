from typing import List
import torch
import torch.nn as nn
from DLIP.models.zoo.compositions.unet_base import UnetBase
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from skimage.segmentation import watershed
from skimage import measure

import numpy as np

class UnetInstance(UnetBase):
    def __init__(
        self,
        in_channels: int,
        loss_fcn: nn.Module,
        encoder_type = 'unet',
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        decoder_type = 'unet',
        dropout: float = 0.0,
        ae_mode = False,
        pretraining_weights = 'imagenet',
        encoder_frozen=False,
        **kwargs,
    ):
        out_channels = 1
        super().__init__(
                in_channels,
                out_channels,
                loss_fcn,
                encoder_type,
                encoder_filters,
                decoder_filters,
                decoder_type,
                dropout,
                ae_mode,
                pretraining_weights,
                encoder_frozen,
                **kwargs)
        self.post_pro = DistMapPostProcessor(**split_parameters(kwargs, ["post_pro"])["post_pro"])
        
    def training_step(self, batch, batch_idx):
        x, y_true   = batch
        y_true      = y_true.permute(0, 3, 1, 2)
        y_pred      = self.forward(x)
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        metric = calc_instance_metric(y_true,y_pred, self.post_pro)
        self.log("val/loss", 1-metric, prog_bar=True, on_epoch=True)
        return  1-metric

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        metric = calc_instance_metric(y_true,y_pred, self.post_pro)
        self.log("test/aji+", metric, prog_bar=True, on_epoch=True, on_step=False)
        return metric

def calc_instance_metric(y_true,y_pred, post_pro):
    metric = list()
    for i_b in range(y_true.shape[0]):
        seeds   = measure.label(y_true[i_b,0,:].cpu().numpy()>0.6, background=0)
        masks   = y_true[i_b,0,:].cpu().numpy()>0.0
        gt_mask = watershed(image=-y_true[i_b,0,:].cpu().numpy(), markers=seeds, mask=masks, watershed_line=False)
        pred_mask = post_pro.process(y_pred[i_b,0,:].cpu().numpy(),None)
        try:
            metric.append(get_fast_aji_plus(remap_label(gt_mask),remap_label(pred_mask)))
        except:
            metric.append(0)
    return np.mean(metric)