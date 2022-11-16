import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import label2rgb
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, grey_closing, generate_binary_structure
from skimage import measure
from skimage.morphology import disk
import tifffile
from difflib import get_close_matches
from skimage.util import crop
import filetype


class CropGenerator:
    def __init__(
            self, 
            root_dir,
            samples_dir = "samples",
            labels_dir = "labels") -> None:
        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.labels_dir = labels_dir
        self.data_dirs = ["train", "test"]

    def generate_cropped_dataset(self, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        for data_dir in self.data_dirs:
            if not os.path.exists(os.path.join(dst_dir, data_dir)):
                os.makedirs(os.path.join(dst_dir, data_dir))
            
            if not os.path.exists(os.path.join(dst_dir, data_dir, self.samples_dir)):
                os.makedirs(os.path.join(dst_dir, data_dir, self.samples_dir))

            if not os.path.exists(os.path.join(dst_dir, data_dir, self.labels_dir)):
                os.makedirs(os.path.join(dst_dir, data_dir, self.labels_dir))

            for sample_file in os.listdir(os.path.join(self.root_dir, data_dir, self.samples_dir)):
                sample_file_path = os.path.join(self.root_dir, data_dir, self.samples_dir, sample_file)
                if not filetype.is_image(sample_file_path):
                    continue
                label_file = get_close_matches(
                    sample_file, 
                    os.listdir(os.path.join(self.root_dir, data_dir, self.labels_dir)))[0]

                label_file_path = os.path.join(self.root_dir, data_dir, self.labels_dir, label_file)

                sample_dst_path = os.path.join(dst_dir, data_dir, self.samples_dir, sample_file)
                label_dst_path  = os.path.join(dst_dir, data_dir, self.labels_dir, label_file)

                sample_raw = tifffile.imread(sample_file_path)
                label_raw = tifffile.imread(label_file_path)

                sample_crop_lst, label_crop_lst = self._generate_crops(sample_raw, label_raw)

                for i in range(len(sample_crop_lst)):
                    tifffile.imwrite(sample_dst_path.replace(".tif",f"_{str(i).zfill(3)}.tif"),sample_crop_lst[i])
                    tifffile.imwrite(label_dst_path.replace("_label.tif",f"_{str(i).zfill(3)}_label.tif"),label_crop_lst[i])

                # print(crop_lst.shape)

                # plt.imshow(sample_raw)
                # plt.show()

                # plt.imshow(label_raw[0,:,:])
                # plt.show()
                # file_path_src = os.path.join(data_dir, self.label_dir, file)
                # file_path_dst = os.path.join(data_dir, folder_name, file)

                # print(file)

                # label_raw = cv2.imread(file_path_src,-1)
                # print(label_raw.shape)

                # label_dist = self._dist_trafo( label_raw)
                # #label_rgb = label2rgb(label_raw[:,:,0],bg_label=0)

                # tifffile.imwrite(file_path_dst.replace("png","tif"),label_dist.astype(np.float32))
                # plt.imshow(label_dist)
                # plt.show()

    def _generate_crops(self, sample,label, size=360):
        N = size
        M =  size

        sample_crops = list()
        label_crops = list()

        for x in range(0,size*(sample.shape[0]//size),M):
            for y in range(0,size*(sample.shape[1]//size),N):
                sample_crop = sample[x:x+M,y:y+N] 
                label_crop = label[:,x:x+M,y:y+N]

                # if np.max(label_crop)>0:
                #     # sample_crops.append(sample_crop)
                #     # label_crops.append(label_crop)

                sample_crops.append(sample_crop)
                label_crops.append(label_crop)

        return sample_crops, label_crops




if __name__ == "__main__":
    root_dir = "/home/ws/sc1357/data/2022_DMA_BF_CPS_Joaquin"

    dst_dir = "/home/ws/sc1357/data/2022_DMA_BF_CPS_Joaquin_360"

    dmg_obj = CropGenerator(root_dir)
    dmg_obj.generate_cropped_dataset(dst_dir)