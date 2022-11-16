import torch
import wandb
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import resnet50

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class SResUnet(pl.LightningModule):
    """Shallow Unet with ResNet18 or ResNet34 encoder.
    """

    def __init__(
        self, loss_fcn: nn.Module,
        in_channels: int = 3,
        out_channels=2,
        input_height=None,
        ae_mode=False,
        imagenet_pretraing=True,
        **kwargs
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.ae_mode = ae_mode
        self.encoder = resnet50(pretrained=imagenet_pretraing)
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(2048, 512)
        self.conv6 = DoubleConv(512 + (256 if not self.ae_mode else 0), 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = DoubleConv(256 + (128 if not self.ae_mode else 0), 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = DoubleConv(128 + (64 if not self.ae_mode else 0), 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = DoubleConv(64 + (64 if not self.ae_mode else 0), 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1) # outconv


    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_conv6(block5)
        if not self.ae_mode:
            x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        if not self.ae_mode:
            x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        if not self.ae_mode:
            x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        if not self.ae_mode:
            x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x

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


    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
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
        if batch_idx == 0:
            self.log_imgs(x,y_pred)
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