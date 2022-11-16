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

LABEL_MODE = 'GENERAL FINDING'

class NIHChestXrayDataset(BaseDataset):
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
            key=lambda x: (int(x.split('/')[-1].split('.')[0].split('_')[0]),int(x.split('/')[-1].split('.')[0].split('_')[1])),
        )
        
        if labels_available:
            patient_id_dict = pd.read_csv(os.path.join(root_dir,'labels.csv')).set_index('Image Index').to_dict()['Finding Labels']
            new_samples_sorted = []
            labels = {}
            for sample in all_samples_sorted:
                if sample.split('/')[-1] in patient_id_dict.keys():
                    new_samples_sorted.append(sample)
                    finding = patient_id_dict[sample.split('/')[-1]]
                    label = -1
                    if LABEL_MODE == 'GENERAL FINDING':
                        label = 0 if finding == 'No Finding' else 1
                    labels[sample.split('/')[-1]] = label
            self.labels = labels
            all_samples_sorted = new_samples_sorted
            all_samples_sorted = sorted(
                all_samples_sorted,
                key=lambda x: (int(x.split('/')[-1].split('.')[0].split('_')[0]),int(x.split('/')[-1].split('.')[0].split('_')[1])),
            )

        self.indices = []
        if not empty_dataset:
            self.indices = [i.split('/')[-1] for i in all_samples_sorted]
        self.raw_mode = False
 
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,f"{self.indices[idx]}"),-1)
        )
        if len(sample_img.shape) != 2:
            sample_img = cv2.cvtColor(sample_img,cv2.COLOR_RGBA2GRAY)
        sample_img = np.expand_dims(sample_img,2)

        if LABEL_MODE == 'GENERAL FINDING':
            label = self.labels[self.indices[idx]]

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
