from typing import List
import torch
import wandb
import torch.nn as nn
import numpy as np

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.resnet_encoder import ResNetEncoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder

from torch.nn.modules.container import ModuleList
import torchvision

class Flatten(nn.Module):
    
    def forward(self,x):
        return torch.flatten(x, 1)

class ResnetClassifier(BaseComposition):
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_fcn: nn.Module,
        encoder_type: str,
        pretraining_weights = 'imagenet',
        encoder_frozen=False,
        class_dict = None,
        **kwargs,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        encoder_type = encoder_type.lower()
        encoder = ResNetEncoder(
            input_channels = in_channels,
            encoder_type = encoder_type,
            pretraining_weights=pretraining_weights,
            encoder_frozen=encoder_frozen,
            classification_output = True
        )
        self.encoder_frozen = encoder_frozen
        self.append(encoder)
        self.append(nn.AdaptiveAvgPool2d((1,1)))
        self.append(Flatten())
        self.append(nn.Linear(in_features=2048,out_features=num_classes))
        self.append(nn.Sigmoid() if num_classes == 1 else nn.Softmax())
        self.num_classes = num_classes
        self.class_dict = class_dict
        if self.class_dict is not None:
            new_class_keys = dict()
            for key in class_dict.keys():
                value = class_dict[key]
                new_class_keys[str(key)] = value
            self.class_dict = new_class_keys

 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)  # shape NxC
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if self.num_classes == 1:
            self.log('val/accuracy',float(sum(((y_pred > 0.5)*1.) == y_true) / len(y_true)), prog_bar=True, on_epoch=True, on_step=False)
        else:
            max_pred = torch.argmax(y_pred,dim=1)
            max_true = torch.argmax(y_true,dim=1)
            for i in range(self.num_classes):
                class_i = torch.where(max_true==i)[0]
                accuracy = torch.sum(max_true[class_i] == max_pred[class_i]) / len(class_i)
                if self.class_dict is None:
                    self.log(f'val/accuracy_{i}',accuracy, prog_bar=True, on_epoch=True, on_step=False)
                else:
                    self.log(f'val/accuracy_{self.class_dict[str(i)]}',accuracy, prog_bar=True, on_epoch=True, on_step=False)
            
            
        if batch_idx == 0:
            self.log_imgs(x,y_pred,y_true)
            #self.xs = x
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)
        if self.num_classes == 1:
                self.log('test/accuracy',float(sum(((y_pred > 0.5)*1.) == y_true) / len(y_true)), prog_bar=True, on_epoch=True, on_step=False)
        else:
            max_pred = torch.argmax(y_pred,dim=1)
            max_true = torch.argmax(y_true,dim=1)
            for i in range(self.num_classes):
                class_i = torch.where(max_true==i)[0]
                accuracy = torch.sum(max_true[class_i] == max_pred[class_i]) / len(class_i)
                if self.class_dict is None:
                    self.log(f'test/accuracy_{i}',accuracy, prog_bar=True, on_epoch=True, on_step=False)
                else:
                    self.log(f'test/accuracy_{self.class_dict[str(i)]}',accuracy, prog_bar=True, on_epoch=True, on_step=False)
            

        return loss


    def log_imgs(self,x,y_pred,y_true,max_imgs=10):
        x_wandb = []
        for i in range(min([max_imgs,len(x)])):
            x_item = x[i].permute(1,2,0).cpu().detach().numpy()
            if self.num_classes > 1:
                x_wandb.append(wandb.Image(x_item,caption=f'Pred: {self.class_dict[str(torch.argmax(y_pred[i]).cpu().numpy())]}, True: {self.class_dict[str(torch.argmax(y_true[i]).cpu().numpy())]}'))
            else:
                x_wandb.append(wandb.Image(x_item,caption=f'Pred: {list(np.round(y_pred[i].cpu().numpy(),2))}, True: {y_true[i].cpu().numpy()}'))
        wandb.log({
            "x": x_wandb,
        })

    def log_grad_cams(self,x,max_imgs=10):
        #cam = GradCAM(model=self, target_layers=[self.composition[-5].backbone[-1][-1]], use_cuda=True)
        cam = GradCAM(model=self, target_layers=[self.composition[-2].layer4[-1]], use_cuda=True)
        imgs = []
        grads = []
        for i in range(min([max_imgs,len(x)])):
            input_tensor = x[i].detach()
            grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(input_tensor.permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=False)
            grads.append(wandb.Image(visualization))
            imgs.append(wandb.Image(input_tensor.permute(1,2,0).detach().cpu().numpy()*255))
        wandb.log({
        'grad':grads
        })