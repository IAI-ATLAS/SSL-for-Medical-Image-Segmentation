import glob
import os
import cv2
import random
from matplotlib.pyplot import cla
import torch
import numpy as np
import pandas as pd
import albumentations as A
import random

from DLIP.data.base_classes.base_dataset import BaseDataset

class PnuemoniaXrayDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="png",
        transforms=None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
    ):
        self.root_dir = root_dir
        self.samples_data_format = samples_data_format
        self.samples_dir = os.path.join(self.root_dir)
        self.labels_available = labels_available
        self.return_trafos = return_trafos
        self.transforms = transforms
        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        all_samples_sorted = glob.glob(f"{self.samples_dir}{os.path.sep}PNEUMONIA/*") + glob.glob(f"{self.samples_dir}{os.path.sep}NORMAL/*")
        self.indices = []
        if not empty_dataset:
            self.indices = all_samples_sorted
 
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_img = np.array(cv2.imread(f"{self.indices[idx]}",-1))
        if len(sample_img.shape) != 2:
            sample_img = cv2.cvtColor(sample_img,cv2.COLOR_RGBA2GRAY)
        sample_img = np.expand_dims(sample_img,2)
        label = 0 if self.indices[idx].split('/')[-2] == 'NORMAL' else 1

        sample_img_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, np.zeros_like(sample_img))
            sample_img_lst.append(im)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
        
        if self.labels_available:
            return sample_img_lst, np.array([label]).astype(np.float32)
        if not self.labels_available:
            return sample_img_lst
        
        
    def edit_sample_size(self,height,width):
        augs = self.transforms[0].transform['aug'][:]
        for i in range(len(augs)):
            if type(augs[i]) == A.RandomResizedCrop:
                augs[i] = A.RandomResizedCrop(
                    always_apply=augs[i].always_apply,
                    p=augs[i].p,
                    height=height,
                    width=width,
                    scale=augs[i].scale,
                    ratio=augs[i].ratio,
                    interpolation=augs[i].interpolation
                )
        self.transforms[0].transform['aug'] = A.Compose(augs)
        augs = self.transforms[0].transform['pre'][:]
        for i in range(len(augs)):
            if type(augs[i]) == A.Resize:
                augs[i] = A.Resize(
                    height=height,
                    width=width,
                )
        self.transforms[0].transform['pre'] = A.Compose(augs) 
        # 1
        augs = self.transforms[1].transform['aug'][:]
        for i in range(len(augs)):
            if type(augs[i]) == A.RandomResizedCrop:
                augs[i] = A.RandomResizedCrop(
                    always_apply=augs[i].always_apply,
                    p=augs[i].p,
                    height=height,
                    width=width,
                    scale=augs[i].scale,
                    ratio=augs[i].ratio,
                    interpolation=augs[i].interpolation
                )
        self.transforms[1].transform['aug'] = A.Compose(augs)
        
        augs = self.transforms[1].transform['pre'][:]
        for i in range(len(augs)):
            if type(augs[i]) == A.Resize:
                augs[i] = A.Resize(
                    height=height,
                    width=width,
                )
        self.transforms[1].transform['pre'] = A.Compose(augs)        

    def get_samples(self):
        return self.indices

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        return self.indices.append(new_sample)
