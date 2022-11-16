import glob
import os
import cv2
import random
from matplotlib.pyplot import cla
import torch
import numpy as np
import pandas as pd
import albumentations as A

from DLIP.data.base_classes.base_dataset import BaseDataset

class CocoDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="jpg",
        transforms=None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        classifier_mode = False,
    ):
        self.root_dir = root_dir
        self.samples_data_format = samples_data_format
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.labels_available = labels_available
        self.return_trafos = return_trafos
        self.transforms = transforms
        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        all_samples_sorted = sorted(
            glob.glob(f"{self.samples_dir}{os.path.sep}*.{self.samples_data_format}"),
            key=lambda x: int(x.split('/')[-1].split('.')[0]),
        )

        self.indices = []
        if not empty_dataset:
            self.indices = [i.split('/')[-1].split('.')[0] for i in all_samples_sorted]
        self.raw_mode = False
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,f"{self.indices[idx]}.{self.samples_data_format}"),cv2.IMREAD_COLOR)
        )
        if sample_img.dtype != 'uint8':
            print()
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        sample_img_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, np.zeros_like(sample_img))
            sample_img_lst.append(im)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
        
        # sample_img_lst (optional: labels) (optional: trafos)
        if self.labels_available:
            return sample_img_lst, 0
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
