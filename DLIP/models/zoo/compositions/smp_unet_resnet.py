
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import wandb
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class UnetDecoderAE(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, 0, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, None)

        return x



class SmpUnetResnet(smp.Unet, pl.LightningModule):

    def __init__(
        self,
        loss_fcn: nn.Module,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        #decoder_channels = (256, 128, 64, 32, 16),
        decoder_channels = (512,256,128,64,32),
        decoder_attention_type  = None,
        in_channels: int = 3,
        out_channels: int = 3,
        activation = None,
        aux_params = None,
        input_height=None,
        ae_mode=False,
        imagenet_pretraing=True
    ):
        super(SmpUnetResnet, self).__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights="imagenet" if imagenet_pretraing else None, 
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=out_channels,
            activation=activation,
            aux_params=aux_params
        )
        self.loss_fcn = loss_fcn
        self.ae_mode = ae_mode
        if ae_mode:
            self.decoder.forward = self.ae_forward
            self.decoder = UnetDecoderAE(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )

    def set_optimizers(self, optimizer, lrs=None, metric_to_track=None):
        self.optimizer = optimizer
        self.lrs = lrs
        self.metric_to_track = metric_to_track
        if self.metric_to_track  is None:
            self.metric_to_track = "val/loss"

    def configure_optimizers(self):
        if self.lrs is None and self.metric_to_track is None:
            return {"optimizer": self.optimizer}
        if self.lrs is None:
            return {"optimizer": self.optimizer, "monitor": self.metric_to_track}
        if self.metric_to_track is None:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lrs}
        return {"optimizer": self.optimizer,"lr_scheduler": self.lrs,"monitor": self.metric_to_track}

    def get_progress_bar_dict(self):
        # don't show the running loss (very iritating)
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    def ae_forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_imgs(x,y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def log_imgs(self,x,y):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        wandb.log({
            "x": x_wandb,
            "y": y_wandb
        })