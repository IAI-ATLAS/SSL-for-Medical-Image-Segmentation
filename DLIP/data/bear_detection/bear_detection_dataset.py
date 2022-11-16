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

class BearDetectionDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="jpg",
        labels_data_format="png",
        transforms=None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        classifier_mode = False,
    ):
        self.root_dir = root_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.labels_dir = os.path.join(self.root_dir, 'labels')
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

        self.classifier_mode = classifier_mode
        self.classification_classes = (pd.read_csv(os.path.join(root_dir,'ground_truth_labels.csv')).set_index('image_id')).to_dict('image_id')
        self.classes_one_hot = {}
        for key in self.classification_classes:
            clazz = self.classification_classes[key]['class']
            class_one_hot = np.zeros(7)
            class_one_hot[clazz] = 1.
            self.classes_one_hot[f'{key:012d}'] = class_one_hot
        self.indices = []
        if not empty_dataset:
            self.indices = [i.split('/')[-1].split('.')[0] for i in all_samples_sorted]
        self.raw_mode = False
 
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if self.classifier_mode:
            return self.getitem_classification(index)
        else:
            return self.getitem_semantic_segmentation(index)
        
        
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

    def getitem_classification(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,f"{self.indices[idx]}.{self.samples_data_format}"),-1)
        )
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        #label = np.expand_dims(np.array((self.classes_one_hot[self.indices[idx]])).astype(np.float32),0)
        label = np.array((self.classes_one_hot[self.indices[idx]])).astype(np.float32)

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

    def getitem_semantic_segmentation(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,f"{self.indices[idx]}.{self.samples_data_format}"),-1)
        )
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        
        label = None
        if self.labels_available:
            label_path = os.path.join(self.labels_dir,f"{self.indices[idx]}.{self.labels_data_format}")
            label = cv2.imread(label_path, -1)
            label = np.where(label == 0, 0,1)
            label = np.expand_dims(label,2)

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label
            else:
                return sample_img
        
        sample_img_lst = []
        label_lst = []
        trafo_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, label)
            sample_img_lst.append(im)
            label_lst.append(lbl)
            trafo_lst.append(trafo)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            label_lst = label_lst[0] if len(label_lst) > 0 else label_lst
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
            
            
        # custom segement out image with mask
        # selected_image = 0
        # if random.random() > 0.5:
        #     selected_image=1
        # rand_val = random.random()
        # if 0.4 > rand_val > 0.0:
        #     # just mask it out black
        #     masked = torch.Tensor(np.where((np.repeat(label_lst[selected_image].numpy(),3,axis=2) == 1),sample_img_lst[selected_image].permute(1,2,0),0)).permute(2,0,1)
        #     sample_img_lst[selected_image] = masked
        # if 0.8 > rand_val > 0.4:
        #     blurred_image = cv2.GaussianBlur(sample_img_lst[selected_image].permute(1,2,0).numpy(), (101, 101), cv2.BORDER_DEFAULT)
        #     masked = torch.Tensor(np.where((np.repeat(label_lst[selected_image].numpy(),3,axis=2) == 1),sample_img_lst[selected_image].permute(1,2,0),blurred_image)).permute(2,0,1)  
        #     sample_img_lst[selected_image] = masked
        #cv2.imwrite('sample.png',(sample_img_lst[selected_image]*255).permute(1,2,0).numpy().astype(np.uint8))
        #cv2.imwrite('masked.png',(masked*255).permute(1,2,0).numpy().astype(np.uint8))
        # end
        
        # sample_img_lst (optional: labels) (optional: trafos)
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if self.return_trafos and not self.labels_available:
            return sample_img_lst, trafo_lst
        if not self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst.float() if isinstance(label_lst,torch.Tensor) else label_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst.float() if isinstance(label_lst,torch.Tensor) else label_lst, trafo_lst
    
    def get_samples(self):
        return self.indices

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        return self.indices.append(new_sample)
