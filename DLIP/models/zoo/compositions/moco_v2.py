"""Adapted from: https://github.com/facebookresearch/moco.
Original work is: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
This implementation is: Copyright (c) PyTorch Lightning, Inc. and its affiliates. All Rights Reserved
This implementation is licensed under Attribution-NonCommercial 4.0 International;
You may not use this file except in compliance with the License.
You may obtain a copy of the License from the LICENSE file present in this folder.

Snatched from: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py

"""
import DLIP
from DLIP.models.zoo.compositions.base_composition import BaseComposition
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from torch import nn
from torch.nn import functional as F
import torchvision
from pl_bolts.metrics import precision_at_k
from DLIP.models.zoo.necks.densecl_neck import DenseCLNeck

from DLIP.models.zoo.necks.moco_v2_neck import Mocov2Neck

class Mocov2(BaseComposition):
    """PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    """
    def __init__(
        self,
        base_encoder = 'resnet50',
        emb_dim: int = 128,
        num_negatives: int = 34607, # queue length for negative training samples
        num_negatives_val: int = 8655, # queue length for negative validation samples 
        encoder_momentum: float = 0.999, # moco momentum of updating key encoder 
        softmax_temperature: float = 0.07,
        neck='moco',
        **kwargs
    ):

        super().__init__()
        self.num_negatives = num_negatives
        self.num_negatives_val = num_negatives_val
        
        self.softmax_temperature = softmax_temperature
        self.encoder_momentum = encoder_momentum

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder,emb_dim)

        neck = neck.lower()
        if neck == 'moco':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Identity()
            #self.encoder_q.avgpool = nn.Identity()
            self.encoder_q = nn.Sequential(self.encoder_q, Mocov2Neck(dim_mlp=dim_mlp,emb_dim=emb_dim))
            self.encoder_k.fc = nn.Identity()
            #self.encoder_k.avgpool = nn.Identity()
            self.encoder_k = nn.Sequential(self.encoder_k, Mocov2Neck(dim_mlp=dim_mlp,emb_dim=emb_dim))
        elif neck == 'densecl':
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Identity()
            self.encoder_q.avgpool = nn.Identity()
            self.encoder_q = nn.Sequential(self.encoder_q, DenseCLNeck(dim_mlp=dim_mlp,emb_dim=emb_dim))
            self.encoder_k.fc = nn.Identity()
            self.encoder_k.avgpool = nn.Identity()
            self.encoder_k = nn.Sequential(self.encoder_k, DenseCLNeck(dim_mlp=dim_mlp,emb_dim=emb_dim))


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(emb_dim, num_negatives_val))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)
        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))


        # instance queue
        self.register_buffer("instance_queue", torch.randn(emb_dim, num_negatives*16)) # hardcode -> to change
        self.instance_queue = nn.functional.normalize(self.instance_queue, dim=0)
        self.register_buffer("instance_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance validation queue
        self.register_buffer("instance_val_queue", torch.randn(emb_dim, num_negatives_val*16)) # hardcode -> to change
        self.instance_val_queue = nn.functional.normalize(self.instance_val_queue, dim=0)
        self.register_buffer("instance_val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder,emb_dim):
        """Override to add your own encoders."""

        if hasattr(torchvision.models,base_encoder):
            template_model = getattr(torchvision.models, base_encoder)
        elif hasattr(DLIP.models.zoo.compositions,base_encoder):
            template_model = getattr(DLIP.models.zoo.compositions, base_encoder)
        encoder_q = template_model(num_classes=emb_dim)
        encoder_k = template_model(num_classes=emb_dim)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr,instance_step, queue,val_step=False):
        # gather keys before updating queue
        if self._use_ddp_or_ddp2(self.trainer):
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if queue[:, ptr : ptr + batch_size].shape[1] < keys.T.shape[1]:
            # queue overflow: add items until end and rest to start
            remaining_items_before_end = queue[:, ptr : ptr + batch_size].shape[1]
            queue[:, ptr : ptr + batch_size] = keys.T[:,:remaining_items_before_end]
            start_point = (ptr+batch_size) - queue.shape[1]
            queue[:, 0 : start_point] = keys.T[:,remaining_items_before_end:]
        else:
            queue[:, ptr : ptr + batch_size] = keys.T
        
        if instance_step:
            if not val_step:
                ptr = (ptr + batch_size) % (self.num_negatives*16)  # move pointer
            else:
                ptr = (ptr + batch_size) % (self.num_negatives_val*16)  # move pointer
        else:
            if not val_step:
                ptr = (ptr + batch_size) % self.num_negatives  # move pointer
            else:
                ptr = (ptr + batch_size) % self.num_negatives_val  # move pointer
        queue_ptr[0] = ptr

    def forward(self, img_q, img_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k

    def training_step(self, batch, batch_idx):
        img_3, img_4 = None, None
        if len(batch[0]) == 4: # instance case. hacky, sorry.
            (img_1,img_2,img_3,img_4), (_) = batch
            img_3 = img_3.flatten(0, 1)
            img_4 = img_4.flatten(0, 1)
        else:
            (img_1,img_2), (_) = batch

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr,instance_step=False)  # dequeue and enqueue
        loss_global = F.cross_entropy(output.float(), target.long())
        self.log("train/loss_global", loss_global, prog_bar=True)
        
        if img_3 is not None: # instance case
            output, target, keys = self(img_q=img_3, img_k=img_4, queue=self.instance_queue)
            self._dequeue_and_enqueue(keys, queue=self.instance_queue, queue_ptr=self.instance_queue_ptr,instance_step=True)  # dequeue and enqueue
            loss = F.cross_entropy(output.float(), target.long())
            self.log("train/loss_instance", loss, prog_bar=True)
            loss = loss_global + loss
            self.log('train/loss', loss, prog_bar=True)
            return loss
        return loss_global

    def validation_step(self, batch, batch_idx):
        img_3, img_4 = None, None
        if len(batch[0]) == 4: # instance case. hacky, sorry.
            (img_1,img_2,img_3,img_4), (_) = batch
            img_3 = img_3.flatten(0, 1)
            img_4 = img_4.flatten(0, 1)
        else:
            (img_1,img_2), (_) = batch

        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr,val_step=True,instance_step=False)  # dequeue and enqueue
        loss_global = F.cross_entropy(output, target.long())
        self.log("val/loss_global", loss_global, prog_bar=True, on_epoch=True)
        
        if img_3 is not None: # instance case
            output, target, keys = output, target, keys = self(img_q=img_3, img_k=img_4, queue=self.instance_queue)
            self._dequeue_and_enqueue(keys, queue=self.instance_val_queue, queue_ptr=self.instance_val_queue_ptr,instance_step=True,val_step=True)  # dequeue and enqueue
            loss = F.cross_entropy(output.float(), target.long())
            self.log("val/loss_instance", loss, prog_bar=True)
            loss = loss_global + loss
            self.log('val/loss', loss, prog_bar=True)
            return loss
        self.log("val/loss", loss_global, prog_bar=True, on_epoch=True)
        return loss_global

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
