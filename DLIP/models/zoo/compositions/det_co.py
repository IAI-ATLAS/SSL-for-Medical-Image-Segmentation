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
from DLIP.models.zoo.compositions.moco_v2 import Mocov2
from DLIP.models.zoo.necks.densecl_neck import DenseCLNeck

from DLIP.models.zoo.necks.moco_v2_neck import Mocov2Neck

class DetCo(Mocov2):
    """PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    """
    
    def __init__(
        self,
        base_encoder='resnet50_DetCo',
        emb_dim: int = 128,
        num_negatives: int = 34607,
        num_negatives_val: int = 8655,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        neck='',
        loss_weights = [0.1, 0.4, 0.7, 1.0],
        **kwargs
        ):
        super().__init__(base_encoder, emb_dim, num_negatives, num_negatives_val, encoder_momentum, softmax_temperature, neck)
        self.loss_weights = loss_weights
        
        # create the queue
        # we have 4 layers and local and global filters 4*2 = 8 reps per sample
        reps_per_sample = 8
        self.register_buffer("queue", torch.randn(emb_dim,reps_per_sample, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(emb_dim, reps_per_sample, num_negatives_val))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)
        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # instance queue
        self.register_buffer("instance_queue", torch.randn(emb_dim, reps_per_sample, num_negatives*16)) # hardcode -> to change
        self.instance_queue = nn.functional.normalize(self.instance_queue, dim=0)
        self.register_buffer("instance_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance validation queue
        self.register_buffer("instance_val_queue", torch.randn(emb_dim, reps_per_sample, num_negatives_val*16)) # hardcode -> to change
        self.instance_val_queue = nn.functional.normalize(self.instance_val_queue, dim=0)
        self.register_buffer("instance_val_queue_ptr", torch.zeros(1, dtype=torch.long))
    
    
    def forward(self, img_q, img_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """
        queue = queue.clone().detach().permute(1,0,2)

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
        l_pos = torch.einsum('bnw,bnw->bn', [q, k]).unsqueeze(2)
        # negative logits: NxK
        l_neg = torch.einsum('bnw,nwk->bnk', [q, queue])
        # "nc,ck->nk"
        
        # l2g logits
        l_pos_cross = torch.einsum('bnw,bnw->bn', [q[:,4:,:], k[:,:4,:]]).unsqueeze(2)
        l_neg_cross = torch.einsum('bnw,nwk->bnk', [q[:,4:,:], queue[:4,:,:]])
        
        l_pos = torch.cat([l_pos, l_pos_cross], dim=1)
        l_neg = torch.cat([l_neg, l_neg_cross], dim=1)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=2)

        # apply temperature
        logits /= self.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[:2], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue,instance_step,val_step=False):
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if queue[:,:, ptr : ptr + batch_size].shape[2] < keys.T.shape[2]:
            # queue overflow: add items until end and rest to start
            remaining_items_before_end = queue[:,:, ptr : ptr + batch_size].shape[2]
            queue[:,:, ptr : ptr + batch_size] = keys.T[:,:,:remaining_items_before_end]
            start_point = (ptr+batch_size) - queue.shape[2]
            queue[:,:, 0 : start_point] = keys.T[:,:,remaining_items_before_end:]
        else:
            queue[:,:, ptr : ptr + batch_size] = keys.T
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
        losses = [F.cross_entropy(output[:,i,:], target[:,i].long()) for i in range(12)]
        total_loss = sum(loss * self.loss_weights[i%4] for i, loss in enumerate(losses))
        self.log("train/loss", total_loss, prog_bar=True)
        
        if img_3 is not None: # instance case
            output, target, keys = self(img_q=img_3, img_k=img_4, queue=self.instance_queue)
            self._dequeue_and_enqueue(keys, queue=self.instance_queue, queue_ptr=self.instance_queue_ptr,instance_step=True)  # dequeue and enqueue
            losses = [F.cross_entropy(output[:,i,:], target[:,i].long()) for i in range(12)]
            total_loss_2 = sum(loss * self.loss_weights[i%4] for i, loss in enumerate(losses))
            self.log("train/loss_instance", total_loss_2, prog_bar=True)
            return total_loss + total_loss_2
            
    
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        (img_1, img_2), labels = batch

        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr,val_step=True, instance_step=False)  # dequeue and enqueue
        losses = [F.cross_entropy(output[:,i,:], target[:,i].long()) for i in range(12)]
        total_loss = sum(loss * self.loss_weights[i%4] for i, loss in enumerate(losses))

        self.log("val/loss", total_loss, prog_bar=True, on_epoch=True)
        return total_loss