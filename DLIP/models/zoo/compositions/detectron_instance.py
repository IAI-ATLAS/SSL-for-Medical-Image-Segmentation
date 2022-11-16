#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
from typing import List

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    hooks,
)

import torch

import numpy as np

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler, get_default_optimizer_params 
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage



from torchmetrics.detection.map import MAP

from argparse import Namespace
from difflib import get_close_matches
from DLIP.utils.loading.split_parameters import split_parameters
from detectron2.config.config import CfgNode
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label

import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

class Detectron2Instance(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        args = Namespace()
        base_cfg_path = os.path.join(
            __file__[0:__file__.find("models")],
            "experiments", "configurations","detectron",
            "Base-RCNN-FPN-InstSeg.yaml")
        cfg = setup(args, base_cfg_path, **kwargs)
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.model = build_model(self.cfg)

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
        
        with torch.no_grad():
            prediction = self.model(batch)

        if self.cfg["MODEL"]["MASK_ON"]:
            val_metric = calc_instance_metric(batch, prediction)
        else:
            val_metric = calc_object_metric(batch, prediction)

        self.log("val/loss", 1-val_metric)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        
        with torch.no_grad():
            prediction = self.model(batch)

        if self.cfg["MODEL"]["MASK_ON"]:
            test_metric = calc_instance_metric(batch, prediction)
        else:
            test_metric = calc_object_metric(batch, prediction)

        self.log("test/score", test_metric)


    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

def setup(args, base_cfg_path, **kwargs):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # load default cfg
    cfg.merge_from_file(base_cfg_path)

    cfg["SEED"] = int(np.random.get_state()[1][0])

    if "base_lr" in kwargs.keys():
        cfg["SOLVER"]["BASE_LR"] = kwargs["base_lr"]
        del kwargs["base_lr"]

    if "sub_batch_size" in kwargs.keys():
        cfg["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"] = int(kwargs["sub_batch_size"])
        cfg["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"] = int(kwargs["sub_batch_size"])
        del kwargs["sub_batch_size"]

    cfg = merge_cfg_from_param_file(cfg,**kwargs)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def merge_cfg_from_param_file(cfg,**kwargs):
    param_dict = split_parameters(kwargs)
    param_dict = {**param_dict, **param_dict["other"]}
    del param_dict["other"]
    for (key,value) in param_dict.items():
        try:
            main_param_group = get_close_matches(key, [ik.lower() for ik in cfg["MODEL"].keys()])[0].upper()
        except:
            logging.warn(f"Parameter {key} not recognized")
            continue
        if isinstance(cfg["MODEL"][main_param_group], CfgNode):
            for (sub_key, sub_val) in value.items():
                try:
                    org_param_name = get_close_matches(sub_key, [sub_name.lower() for sub_name in cfg["MODEL"][main_param_group].keys()])[0].upper()
                except:
                    logging.warn(f"Parameter {sub_key} not recognized")
                    continue
    
                if isinstance(sub_val, np.ndarray):
                    cfg["MODEL"][main_param_group][org_param_name] = sub_val.tolist()
                else:
                    cfg["MODEL"][main_param_group][org_param_name] = sub_val
        else:
            if isinstance(value, np.ndarray):
                cfg["MODEL"][main_param_group] = value.tolist()
            else:
                cfg["MODEL"][main_param_group] = value          

    return cfg

def calc_object_metric(batch, prediction):
    map = MAP()

    for i_b in range(len(batch)):
        pred = [
            dict(
                boxes = prediction[i_b]["instances"].pred_boxes.tensor.cpu(),
                scores =prediction[i_b]["instances"].scores.cpu(),
                labels =prediction[i_b]["instances"].pred_classes.cpu(),
            )
        ]

        gt = [
            dict(
                boxes = batch[i_b]["instances"].gt_boxes.tensor.cpu(),
                labels= batch[i_b]["instances"].gt_classes.cpu(),
            )
        ]
        map.update(pred, gt)

    res = map.compute()
         
    if res["map"].item()==-1:
        res["map"] = 0

    return res["map"]

def calc_instance_metric(batch, prediction):
    metric = list()
    for i_b in range(len(batch)):
        gt_mask = get_mask_encoding(batch[i_b]["instances"].gt_masks.tensor)
        pred_mask = get_mask_encoding(prediction[i_b]["instances"].pred_masks)
        metric.append(get_fast_aji_plus(remap_label(gt_mask),remap_label(pred_mask)))
        
    return np.mean(metric)

def get_mask_encoding(tensor):
    mask = np.zeros((tensor.shape[1:]), dtype=np.int16)
    for i_i in range(tensor.shape[0]):
        mask[tensor[i_i].detach().cpu().numpy()] =  i_i + 1

    return mask

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