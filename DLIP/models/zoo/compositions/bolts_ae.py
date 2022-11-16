from pl_bolts.models.autoencoders import AE
from pl_bolts.models.autoencoders import VAE
import torch.nn as nn
import torch
import wandb
import pytorch_lightning as pl

class BoltsAE(AE, pl.LightningModule):

    def __init__(
        self,
        input_height: int,
        loss_fcn: nn.Module,
        in_channels: int = 3,
        enc_type: str = 'resnet50',
        first_conv: bool = True,
        maxpool1: bool = True,
        enc_out_dim: int = 2048,
        latent_dim: int = 512,
        lr: float = 0.001,
        out_channels=3,
        ae_mode=False,
        imagenet_pretraing=True,
        **kwargs
    ):
        super(BoltsAE,self).__init__(
            input_height=input_height,
            enc_type=enc_type,
            first_conv=first_conv,
            maxpool1=maxpool1,
            enc_out_dim=enc_out_dim,
            latent_dim=latent_dim,
            lr=lr,
            **kwargs
        )
        self.decoder.conv1 = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.loss_fcn = loss_fcn
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)


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
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        recon_loss = torch.mean(recon_loss)
        self.log("train/loss", recon_loss, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        loss = torch.mean(recon_loss)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0:
            self.log_imgs(x,x_hat)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat = self.forward(x)
        recon_loss = self.loss_fcn(x_hat, x)  # shape NxC
        loss = torch.mean(recon_loss)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def log_imgs(self,x,y,img_limit=3):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        wandb.log({
            "x": x_wandb[:img_limit],
            "y": y_wandb[:img_limit]
        })

