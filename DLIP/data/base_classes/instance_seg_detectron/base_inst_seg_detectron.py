import tifffile
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DLIP.utils.helper_functions.gray_level_check import gray_redundand
import torch
from DLIP.data.base_classes.instance_segmentation.base_inst_seg_dataset import BaseInstanceSegmentationDataset
from detectron2.structures import Instances
from detectron2.structures import Boxes, BitMasks


class BaseInstanceSegmentationDectronDataset(BaseInstanceSegmentationDataset):
    def __getitem__(self, idx):
        # load sample
        sample_path = os.path.join(self.samples, f"{self.indices[idx]}.{self.samples_data_format}")
        if self.samples_data_format=="tif":
            sample_img = tifffile.imread(sample_path)
        else:
            sample_img = cv2.imread(sample_path,-1)
            sample_img = cv2.cvtColor(sample_img,cv2.COLOR_BGR2RGB)
        
        # if sample_img.ndim>2 and gray_redundand(sample_img):
        #     sample_img = sample_img[:,:,0]

        if self.labels_available:
            label_path = os.path.join(self.labels, f"{self.label_prefix}{self.indices[idx]}{self.label_suffix}.{self.labels_data_format}")
            mask_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
            mask_img = mask_img.squeeze().astype(np.int16)

        # # raw mode -> no transforms
        if not self.raw_mode:
            for transform in self.transforms:
                sample_img, mask_img, _ = transform(sample_img, mask_img)

            mask_img = mask_img.numpy()

        obj_ids = np.unique(mask_img)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask_img == obj_ids[:, None, None]
        
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        #print(num_objs)
        # print(num_objs)
        boxes = []
        rel_inst = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax-xmin>1 and ymax-ymin>1:
                rel_inst.append(i)
                
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = [boxes[rel_id] for rel_id in rel_inst]
        masks = masks[rel_inst,:]
        num_objs = len(rel_inst)

        target = Instances(mask_img.shape)
        target.gt_boxes = Boxes(boxes)
        target.gt_classes = torch.zeros((num_objs,), dtype=torch.int64)

        if num_objs>0:
            target.gt_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
                )
        else:
            target.gt_masks = BitMasks(
                    torch.zeros(0,mask_img.shape[0],mask_img.shape[1])
            )
        
        return  {"image": sample_img, "instances": target}