#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
from typing import List
import torch

import detectron2.utils.comm as comm
from detectron2.engine import (
    DefaultTrainer,
    hooks,
)

import torch

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler, get_default_optimizer_params 
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage

from torch.nn import functional as F

from DLIP.objectives import DiceLoss
from DLIP.utils.metrics.seg_metrics import get_dsc

from argparse import Namespace

import pytorch_lightning as pl

from DLIP.models.zoo.compositions.detectron_instance import setup
from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead
from detectron2.config.config import CfgNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

class SemSegFPNHeadAdapted(SemSegFPNHead):
    def losses(self,predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        if predictions.shape[1]==1:
            # loss = F.binary_cross_entropy_with_logits(
            #     predictions, targets.unsqueeze(1), reduction="mean"
            # )
            loss = DiceLoss()(
                            torch.nn.functional.sigmoid(predictions), targets.unsqueeze(1)
                        )
        else:
            loss = F.cross_entropy(
                predictions, targets, reduction="mean", ignore_index=self.ignore_value
            )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


class Detectron2Semantic(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        args = Namespace()
        base_cfg_path = os.path.join(
            __file__[0:__file__.find("models")],
            "experiments", "configurations","detectron",
            "Base-RCNN-FPN-SemSeg.yaml")
        cfg = setup(args, base_cfg_path, **kwargs)
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.model = build_model(self.cfg)
        self.model.sem_seg_head = SemSegFPNHeadAdapted(
            **self.model.sem_seg_head.from_config(self.cfg,self.model.backbone.output_shape())
            )

    def setup(self, stage: str):
        if self.cfg.MODEL.WEIGHTS and not stage=="test":
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                self.cfg.OUTPUT_DIR,
            )
            logger.info(f"Load model weights from checkpoint: {self.cfg.MODEL.WEIGHTS}.")
            # Only load weights, use lightning checkpointing if you want to resume
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)      

    def training_step(self, batch, batch_idx):
        with EventStorage() as storage:
            loss_dict = self.model(batch)

        self.log("train/loss", sum(loss_dict.values()))

        return sum(loss_dict.values())

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        
        prediction = self.model(batch)

        pred    = torch.nn.functional.sigmoid(torch.stack([elem["sem_seg"] for elem in prediction]))
        gt      = torch.stack([elem["sem_seg"] for elem in batch]).unsqueeze(1)
        
        val_metric = get_dsc(pred, gt)
        self.log("val/loss", 1-val_metric)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        
        with torch.no_grad():
            prediction = self.model(batch)

        pred    = torch.nn.functional.sigmoid(torch.stack([elem["sem_seg"] for elem in prediction]))
        gt      = torch.stack([elem["sem_seg"] for elem in batch]).unsqueeze(1)
        
        test_metric = get_dsc(pred, gt)
        self.log("test/score", test_metric)


    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )

    return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )