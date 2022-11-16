import glob
import os
import cv2
import random
from matplotlib.pyplot import cla
import torch
import numpy as np
import pandas as pd

from DLIP.data.base_classes.base_dataset import BaseDataset

class IsicDermoDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_data_format="jpg",
        labels_data_format="png",
        transforms=None,
        empty_dataset=False,
        insert_bg_class=False,
        labels_available=True,
        return_trafos=False,
        classifier_mode = False,
        classify_melanoma = False,
        balance_classes = False
    ):
        self.root_dir = root_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.insert_bg_class = insert_bg_class
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
            key=lambda x: int(x.split(f'.{self.samples_data_format}')[0].split('_')[-1]),
        )

        self.classifier_mode = classifier_mode
        if self.classifier_mode:
            # 0,1 -> seborrheic_keratosis
            # 0,0 -> Unknown / nevus
            # 1,0 -> Melanoma
            # one hot = [Melanoma,seborrheic_keratosis,nevus]
            # temp:   0        1          2
            #       Melanoma,Seborreheic,unknown
            num_melanoma = 0
            num_keratosis = 0
            num_unknown = 0
            classification_classes = (pd.read_csv(os.path.join(root_dir,'ground_truth_labels.csv')).set_index('image_id')).to_dict('image_id')
            self.classification_classes_one_hot = {}
            for key in classification_classes.keys():
                item = classification_classes[key]
                # Melanoma detection 
                if item['melanoma'] == 1.0 and item['seborrheic_keratosis'] == 0.0:
                    self.classification_classes_one_hot[key] = np.array([0]).astype(np.float32)
                if item['melanoma'] == 0.0 and item['seborrheic_keratosis'] == 0.0:
                    self.classification_classes_one_hot[key] = np.array([2]).astype(np.float32)
                if item['melanoma'] == 0.0 and item['seborrheic_keratosis'] == 1.0:
                    self.classification_classes_one_hot[key] = np.array([1]).astype(np.float32)
                
                if item['melanoma'] == 1.0 and item['seborrheic_keratosis'] == 0.0:
                    num_melanoma+=1
                if item['melanoma'] == 0.0 and item['seborrheic_keratosis'] == 0.0:
                    num_unknown+=1
                if item['melanoma'] == 0.0 and item['seborrheic_keratosis'] == 1.0:
                    num_keratosis+=1

        self.indices = []
        if not empty_dataset:
            self.indices = [i.split(f'.{self.samples_data_format}')[0].split('_')[-1] for i in all_samples_sorted]
        self.raw_mode = False
    
    def __len__(self):
        return len(self.indices)
    
    
    def balance_classes(self):
        all_ones = [x for x in self.indices if int(self.classification_classes_one_hot[f'ISIC_{x}']) == 1]
        all_zeros = [x for x in self.indices if int(self.classification_classes_one_hot[f'ISIC_{x}']) == 0]
        multplied = [entry for entry in all_ones for _ in range(round(len(all_zeros) / len(all_ones)))]
        combined = self.indices + multplied
        self.indices = combined

    def __getitem__(self, index):
        if self.classifier_mode:
            return self.getitem_classification(index)
        else:
            return self.getitem_semantic_segmentation(index)

    def getitem_classification(self, idx):
        sample_img = np.array(
            cv2.imread(os.path.join(self.samples_dir,
                f"ISIC_{self.indices[idx]}.{self.samples_data_format}"),
                -1,
            )
        )
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        label = self.classification_classes_one_hot[f'ISIC_{self.indices[idx]}']

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
            cv2.imread(os.path.join(self.samples_dir,
                f"ISIC_{self.indices[idx]}.{self.samples_data_format}"),
                -1,
            )
        )

        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        label = None
        if self.labels_available:
            label_path = os.path.join(
                self.labels_dir, 
                f"ISIC_{self.indices[idx]}_segmentation.{self.labels_data_format}"
            )
            label = cv2.imread(label_path, -1)
            label = np.where(label == 0, 0,1)
            label = np.expand_dims(label,2)
            if self.insert_bg_class:
                # convert to one hot encoding
                label = np.stack((label,np.where(label,0,1)))

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
