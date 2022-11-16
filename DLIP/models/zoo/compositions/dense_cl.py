
from torch import nn
import torch
from torch.nn import functional as F
from pl_bolts.metrics import precision_at_k
from DLIP.models.zoo.compositions.moco_v2 import Mocov2
from DLIP.utils.helper_functions.split_image import slice_image


class DenseCL(Mocov2):

    def __init__(
        self,
        base_encoder='CustomResnet',
        emb_dim: int = 128,
        num_negatives: int = 34607,
        num_negatives_val: int = 8655,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        loss_lambda = 0.5,
        neck='densecl',
        **kwargs
    ):
        super().__init__(base_encoder, emb_dim, num_negatives, num_negatives_val, encoder_momentum, softmax_temperature, neck)

        self.loss_lambda = loss_lambda
        # create the dense queue
        self.register_buffer("dense_queue", torch.randn(emb_dim, num_negatives))
        self.dense_queue  = nn.functional.normalize(self.dense_queue, dim=0)
        self.register_buffer("dense_queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation dense queue
        self.register_buffer("dense_val_queue", torch.randn(emb_dim, num_negatives_val))
        self.dense_val_queue = nn.functional.normalize(self.dense_val_queue, dim=0)
        self.register_buffer("dense_val_queue_ptr", torch.zeros(1, dtype=torch.long))


        # instance 
        # create the dense queue
        self.register_buffer("instance_dense_queue", torch.randn(emb_dim, num_negatives*16))
        self.instance_dense_queue  = nn.functional.normalize(self.instance_dense_queue, dim=0)
        self.register_buffer("instance_dense_queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation dense queue
        self.register_buffer("instance_dense_val_queue", torch.randn(emb_dim, num_negatives_val*16))
        self.instance_dense_val_queue = nn.functional.normalize(self.instance_dense_val_queue, dim=0)
        self.register_buffer("instance_dense_val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, img_q, img_k, queue, dense_queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """
        # N = batch size
        # C = Hidden dim size
        # S = filter size output resnet
        q_b = self.encoder_q[0](img_q) # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
       # q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)
        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            k_b = self.encoder_k[0](img_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            #k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)
            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

        loss = {}
        # CONTRASTIVE 
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
        # logits: Nx(1+K)
        # HEAD
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.softmax_temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        
        loss['contrastive'] = [logits,labels,k]
        
        # DENSECL
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2
        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1
        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid, dense_queue.clone().detach()])
        
        logits_dense = torch.cat([l_pos_dense, l_neg_dense], dim=1)
        logits_dense /= self.softmax_temperature
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long)
        labels_dense = labels_dense.type_as(logits_dense)
        loss['dense'] = [logits_dense,labels_dense,k2]
        
        return loss

    def training_step(self, batch, batch_idx):

        img_3, img_4 = None, None
        if len(batch[0]) == 4: # instance case. hacky, sorry.
            (img_1,img_2,img_3,img_4), (_) = batch
            img_3 = img_3.flatten(0, 1)
            img_4 = img_4.flatten(0, 1)
        else:
            (img_1,img_2), (_) = batch

        self._momentum_update_key_encoder()  # update the key encoder
        loss = self(img_q=img_1, img_k=img_2, queue=self.queue, dense_queue=self.dense_queue)
        
        # contrastive loss
        output, target, keys = loss['contrastive']
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr,instance_step=False)  # dequeue and enqueue
        loss_contrastive = F.cross_entropy(output.float(), target.long())
        
        # dense loss
        output_dense, target_dense, keys_dense = loss['dense']
        self._dequeue_and_enqueue(keys_dense, queue=self.dense_queue, queue_ptr=self.dense_queue_ptr,instance_step=False)
        loss_dense = F.cross_entropy(output_dense.float(), target_dense.long())
        
        loss_global = (loss_contrastive * self.loss_lambda) + (loss_dense * (1-self.loss_lambda))

        self.log("train/loss_contrastive", loss_contrastive, prog_bar=True)
        self.log("train/loss_dense", loss_dense, prog_bar=True)
        self.log("train/loss", loss_global, prog_bar=True)

        if img_3 is not None: # instance case
            loss = self(img_q=img_3, img_k=img_4, queue=self.instance_queue, dense_queue=self.instance_dense_queue)
            # contrastive loss
            output, target, keys = loss['contrastive']
            self._dequeue_and_enqueue(keys, queue=self.instance_queue, queue_ptr=self.instance_queue_ptr,instance_step=True)  # dequeue and enqueue
            loss_contrastive = F.cross_entropy(output.float(), target.long())
            
            # dense loss
            output_dense, target_dense, keys_dense = loss['dense']
            self._dequeue_and_enqueue(keys_dense, queue=self.instance_dense_queue, queue_ptr=self.instance_dense_queue_ptr,instance_step=True)
            loss_dense = F.cross_entropy(output_dense.float(), target_dense.long())
            
            instance_loss = (loss_contrastive * self.loss_lambda) + (loss_dense * (1-self.loss_lambda))

            self.log("train/instance_loss_contrastive", loss_contrastive, prog_bar=True)
            self.log("train/instance_loss_dense", loss_dense, prog_bar=True)
            self.log("train/instance_loss", instance_loss, prog_bar=True)


            return loss_global + instance_loss
        return loss_global

    def validation_step(self, batch, batch_idx):
        img_3, img_4 = None, None
        if len(batch[0]) == 4: # instance case. hacky, sorry.
            (img_1,img_2,img_3,img_4), (_) = batch
            img_3 = img_3.flatten(0, 1)
            img_4 = img_4.flatten(0, 1)
        else:
            (img_1,img_2), (_) = batch
        
        loss = self(img_q=img_1, img_k=img_2, queue=self.val_queue, dense_queue=self.dense_val_queue)
        # contrastive loss
        output, target, keys = loss['contrastive']
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr,val_step=True,instance_step=False)  # dequeue and enqueue
        loss_contrastive = F.cross_entropy(output.float(), target.long())        
        
        # dense loss
        output_dense, target_dense, keys_dense = loss['dense']
        self._dequeue_and_enqueue(keys_dense, queue=self.dense_val_queue, queue_ptr=self.dense_val_queue_ptr,val_step=True,instance_step=False)
        loss_dense = F.cross_entropy(output_dense.float(), target_dense.long())
        
        loss_global = (loss_contrastive * self.loss_lambda) + (loss_dense * (1-self.loss_lambda))

        self.log("val/loss_contrastive", loss_contrastive, prog_bar=True)
        self.log("val/loss_dense", loss_dense, prog_bar=True)
        self.log("val/loss", loss_global, prog_bar=True)

        if img_3 is not None: # instance case
            loss = self(img_q=img_3, img_k=img_4, queue=self.instance_val_queue, dense_queue=self.instance_dense_val_queue)
            # contrastive loss
            output, target, keys = loss['contrastive']
            self._dequeue_and_enqueue(keys, queue=self.instance_val_queue, queue_ptr=self.instance_val_queue_ptr,val_step=True,instance_step=True)  # dequeue and enqueue
            loss_contrastive = F.cross_entropy(output.float(), target.long())
            
            # dense loss
            output_dense, target_dense, keys_dense = loss['dense']
            self._dequeue_and_enqueue(keys_dense, queue=self.instance_dense_val_queue, queue_ptr=self.instance_dense_val_queue_ptr,val_step=True,instance_step=True)
            loss_dense = F.cross_entropy(output_dense.float(), target_dense.long())
            
            instance_loss = (loss_contrastive * self.loss_lambda) + (loss_dense * (1-self.loss_lambda))

            self.log("val/instance_loss_contrastive", loss_contrastive, prog_bar=True)
            self.log("val/instance_loss_dense", loss_dense, prog_bar=True)
            self.log("val/instance_loss", instance_loss, prog_bar=True)

            return loss_global + instance_loss
        return loss_global
