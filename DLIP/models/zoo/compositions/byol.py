from typing import Any, Callable, Optional
from pytorch_lightning.trainer.trainer import Trainer
from torch.optim.optimizer import Optimizer
from argparse import Namespace
import torch.nn as nn
import torch
import torchvision
import DLIP
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from torch.nn import functional as F
from copy import deepcopy

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.objectives.byol_loss import BYOLLoss


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class BYOLProjection(nn.Module):
    def __init__(self, encoder, encoder_out_dim=2048, projector_hidden_size=4096, projector_out_dim=256,**kwargs):
        super().__init__()

        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        # Predictor
        self.predictor = MLP(projector_out_dim, projector_hidden_size, projector_out_dim)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h



class BYOL(BaseComposition):
    
    def __init__(
        self,
        backbone = 'resnet50',
        emb_dim=128,
        loss_fcn = BYOLLoss(),
        **kwargs
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.online_network = BYOLProjection(self.init_encoder(backbone,emb_dim))
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

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
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int):
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch):
        x, y = batch

        if len(x) != 2:
            raise Exception('Wrong length for BYOL')
        
        augmentation_1 = x[0]
        augmentation_2 = x[1]

        y1, z1, h1 = self.online_network(augmentation_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(augmentation_2)
        loss_a = self.loss_fcn(h1,z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(augmentation_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(augmentation_1)
        # L2 normalize
        loss_b = self.loss_fcn(h1,z2)

        # Final loss
        total_loss = loss_a + loss_b

        return total_loss

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 2000,verbose=True)
        return [optim], [scheduler]
