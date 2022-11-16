import glob
import os
from random import sample
import cv2
import torch
import numpy as np

from DLIP.data.base_classes.base_dataset import BaseDataset

class CatsVsDogsDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="jpg",
        transforms=None,
        empty_dataset=False,
        labels_available=True,
    ):
        self.root_dir = root_dir
        self.samples_data_format = samples_data_format
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.labels_available = labels_available
        self.transforms = transforms
        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]
        if empty_dataset:
            self.indices = []
        else:
            self.indices = sorted(
                glob.glob(f"{self.samples_dir}{os.path.sep}*.{self.samples_data_format}"),
                key=lambda x: int(x.split('.')[-2]),
            )


        # self.indices = []
        # if not empty_dataset:
        #     self.indices = [i.split(f'.{self.samples_data_format}')[0].split('_')[-1] for i in all_samples_sorted]
        
        self.raw_mode = False
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_img = np.array(cv2.imread(self.indices[idx]))
        try:
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        except Exception:
            print()

        label = 0 if self.indices[idx].split('/')[-1].split('.')[0] == 'dog' else 1
        label = np.array([label]).astype(np.float32)

        sample_img_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, np.zeros_like(sample_img))
            sample_img_lst.append(im)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
        
        # sample_img_lst (optional: labels) (optional: trafos)
        if self.labels_available:
            return sample_img_lst, label
        if not self.labels_available:
            return sample_img_lst
    
    def get_samples(self):
        return self.indices

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        return self.indices.append(new_sample)
