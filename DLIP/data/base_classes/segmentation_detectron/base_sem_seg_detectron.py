import tifffile
import os
import numpy as np
import cv2
from DLIP.utils.helper_functions.gray_level_check import gray_redundand
from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

class BaseSemanticSegmentationDectronDataset(BaseSegmentationDataset):
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
            label_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
            label_one_hot = np.zeros((label_img.shape[0],label_img.shape[1]), dtype=np.float32)
            for key, value in self.map_look_up.items():
                label_one_hot[label_img==value] = int(key)+1

        # # raw mode -> no transforms
        if not self.raw_mode:
            for transform in self.transforms:
                sample_img, label_one_hot, _ = transform(sample_img, label_one_hot)

        
        return  {"image": sample_img, "sem_seg": label_one_hot}