from typing import Callable, Optional
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch
import DLIP
from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.objectives.cross_correlation_loss import CrossCorrelationLoss
import torchvision

class BarlowTwinsProjection(nn.Module):
    
    def __init__(self, in_dims = 2048, h_dims = 2048, out_dims = 2048,**kwargs):
        super().__init__()

        l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                        nn.BatchNorm1d(h_dims),
                        nn.ReLU(inplace=True))

        l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                        nn.BatchNorm1d(h_dims),
                        nn.ReLU(inplace=True))

        l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                        nn.BatchNorm1d(out_dims))

        self.model = nn.Sequential(l1, l2, l3)


    def forward(self, x):
        x = self.model(x)
        return x


class BarlowTwins(BaseComposition):
    
    def __init__(
        self,
        backbone = 'resnet50',
        loss_fcn = CrossCorrelationLoss(),
        emb_dim = 128,
        **kwargs
    ):
        super().__init__()
        self.backbone = self.init_encoder(backbone,emb_dim)
        self.projection = BarlowTwinsProjection()
        self.loss_fcn = loss_fcn

    def init_encoder(self, base_encoder,emb_dim):
        """Override to add your own encoders."""

        if hasattr(torchvision.models,base_encoder):
            template_model = getattr(torchvision.models, base_encoder)
        elif hasattr(DLIP.models.zoo.compositions,base_encoder):
            template_model = getattr(DLIP.models.zoo.compositions, base_encoder)
        encoder = template_model(num_classes=emb_dim)
        encoder.fc = nn.Identity()
        #encoder.avgpool = nn.Identity()
        return encoder


    def forward(self, x):
        x = self.backbone(x)
        return x

    def shared_step(self, batch):
        x, y = batch

        if len(x) != 2:
            raise Exception('Wrong length for Barlow Twins')
        
        augmentation_1 = x[0]
        augmentation_2 = x[1]

        # get h representations, bolts resnet returns a list
        h1 = self(augmentation_1)
        h2 = self(augmentation_2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return self.loss_fcn(z1, z2)

    def training_step(self, batch, batch_idx):
        if len(batch[0]) == 2:
            loss = self.shared_step(batch)
            self.log("train/loss", loss, prog_bar=True)
            return loss
        elif len(batch[0]) == 4:
            loss_global = self.shared_step([batch[0][:2],batch[1][:2]])
            self.log("train/loss_global", loss_global, prog_bar=True)
            loss_instance = self.shared_step([[batch[0][2].flatten(0, 1),batch[0][3].flatten(0, 1)],batch[1][2:]])
            self.log("train/loss_instance", loss_instance, prog_bar=True)
            self.log("train/loss", loss_global+loss_instance, prog_bar=True)
            return loss_global + loss_instance

    def validation_step(self, batch, batch_idx):
        if len(batch[0]) == 2:
            loss = self.shared_step(batch)
            self.log("val/loss", loss,prog_bar=True, on_epoch=True, on_step=False)
            return loss
        elif len(batch[0]) == 4:
            loss_global = self.shared_step([batch[0][:2],batch[1][:2]])
            self.log("val/loss_global", loss_global, prog_bar=True, on_epoch=True, on_step=False)
            loss_instance = self.shared_step([[batch[0][2].flatten(0, 1),batch[0][3].flatten(0, 1)],batch[1][2:]])
            self.log("val/loss_instance", loss_instance, prog_bar=True, on_epoch=True, on_step=False)
            self.log("val/loss", loss_global+loss_instance, prog_bar=True, on_epoch=True, on_step=False)
            return loss_global + loss_instance

    def test_step(self, batch, batch_idx):
        if len(batch[0]) == 2:
            loss = self.shared_step(batch)
            self.log("test/loss", loss,prog_bar=True, on_epoch=True, on_step=False)
            return loss
        elif len(batch[0]) == 4:
            loss_global = self.shared_step([batch[0][:2],batch[1][:2]])
            self.log("test/loss_global", loss_global, prog_bar=True, on_epoch=True, on_step=False)
            loss_instance = self.shared_step([[batch[0][2].flatten(0, 1),batch[0][3].flatten(0, 1)],batch[1][2:]])
            self.log("test/loss_instance", loss_instance, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test/loss", loss_global+loss_instance, prog_bar=True, on_epoch=True, on_step=False)
            return loss_global + loss_instance

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable],
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool
    ):
        
        # 16 steps ~ 1 epoch
        if epoch <= 10:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 160.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        super().optimizer_step(epoch=epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_idx=optimizer_idx, optimizer_closure=optimizer_closure, on_tpu=on_tpu, using_native_amp=using_native_amp, using_lbfgs=using_lbfgs)


    def configure_optimizers(self):
        param_weights = []
        param_biases = []
        for param in self.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optim = torch.optim.SGD(parameters, lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000,verbose=True)
        return [optim], [scheduler]
