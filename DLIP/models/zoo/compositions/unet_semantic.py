from typing import List
import torch
import torch.nn as nn
from DLIP.models.zoo.compositions.unet_base import UnetBase
import wandb

class UnetSemantic(UnetBase):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
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
        out_channels = num_classes
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
        
        if num_classes==1:
            self.append(nn.Sigmoid())
            sd = torch.load('/home/ws/kg2371/lsdf/Daten_Schilling/trained_ann/2022_07_18_DAL_AE/UNet/dnn_weights.ckpt')['state_dict']
            sd_filtered = {}
            for key in sd.keys():
                if 'decoder' not in key:
                    sd_filtered[key] = sd[key]
            self.load_state_dict(sd_filtered,strict=False)
        else:
            self.append(nn.Softmax())
        
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
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self.log_imgs(x,y_pred)
        return  loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("test/score", 1-loss, prog_bar=True, on_epoch=True, on_step=False)
        return 1-loss

    def log_imgs(self,x,y):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        wandb.log({
            "x": x_wandb,
            "y": y_wandb
        })