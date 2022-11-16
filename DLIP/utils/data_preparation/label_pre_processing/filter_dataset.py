import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from PyQt5.QtWidgets import QMessageBox



class DistanceMapGenerator:
    def __init__(
            self, 
            src_dir,
            sample_dir = "samples",
            label_dir = "labels") -> None:
        self.src_dir = src_dir
        self.sample_dir = sample_dir
        self.label_dir = label_dir


    def generate_filtered_dataset(self, dst_dir, folder_name):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if not os.path.exists(os.path.join(dst_dir,folder_name)):
            os.makedirs(os.path.join(dst_dir,folder_name))
            os.makedirs(os.path.join(dst_dir,folder_name,self.sample_dir))
            os.makedirs(os.path.join(dst_dir,folder_name,self.label_dir))


        for file in os.listdir(os.path.join(self.src_dir,folder_name,self.sample_dir)):
            file_path_src = os.path.join(src_dir,folder_name, self.sample_dir,file)
            file_path_dst = os.path.join(dst_dir,folder_name, self.sample_dir,file)

            label_path_src = os.path.join(src_dir,folder_name, self.label_dir,file)
            label_path_dst = os.path.join(dst_dir,folder_name, self.label_dir,file)
            img_raw = cv2.imread(file_path_src)

            print(file_path_src)

            print(img_raw)

            plt.imshow(img_raw)
            plt.show(block=False)

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Message box pop up window")
            msgBox.setWindowTitle("QMessageBox Example")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                shutil.copy(file_path_src,file_path_dst )
                shutil.copy(label_path_src,label_path_dst )

            plt.close()





if __name__ == "__main__":
    src_dir = "/home/ws/sc1357/data/BBBC038"

    dst_dir = "/home/ws/sc1357/data/filtered_BBBC038"

    dmg_obj = DistanceMapGenerator(src_dir)
    dmg_obj.generate_filtered_dataset(dst_dir, "test")