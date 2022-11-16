from pathlib import Path
from tkinter import N
from turtle import forward
import numpy as np
from tqdm import tqdm
import cv2
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from DLIP.models.zoo.compositions.barlow_twins import BarlowTwins


class SegSelector(nn.Module):
    
    def __init__(self,index=0):
        super().__init__()
        self.index = index
        
    def forward(self,x):
        return x[self.index]

def nearest_neighbors(value, array):
    return np.argsort(np.array([np.linalg.norm(value-x) for x in array]))

def farthest_neighbors(value, array):
    return np.argsort(np.array([np.linalg.norm(value-x) for x in array]))[::-1]

def get_nearest_neighbour(num_classes,channels,directory,model,data, class_type='segmentation',nearest=False):
    nbr_neighbors=2
    samples = 80
    latent_dim = None
    sub_dir = 'nearest_neighbours'
    if not nearest:
        sub_dir = 'farthest_neighbours'
    #if num_classes == 1:
    if False:
        print('not implemented ...')
    else:
        Path(f"{directory}/{sub_dir}").mkdir(parents=True, exist_ok=True)
        

        composition = ModuleList()
        if class_type == 'resnet_classifier':
            composition.append(model.composition[0])
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            latent_dim = 2048
        if class_type == 'segmentation':
            composition.append(model.composition[0].cuda())
            composition.append(SegSelector())
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            
        latent_dim = 2048
        #latent_dim = 1028
        embeddings = np.zeros((0,latent_dim))
        y_trues = np.zeros((0,num_classes))
        xs = np.zeros((0,256,256,channels))
        for batch in tqdm(data.test_dataloader()):
            x,y_true = batch
            y_pred = x.to('cuda')
            for item in composition:
                y_pred = item(y_pred)
            y_pred = y_pred.squeeze()
            embeddings = np.concatenate((embeddings,y_pred.cpu().detach()),axis=0)
            xs = np.concatenate((xs,x.permute(0,2,3,1).cpu().detach()*255),axis=0)
            y_trues = np.concatenate((y_trues, y_true.cpu().detach().numpy()),axis=0)
        
        y_trues_str = [''] * len(y_trues)
        for i in range(len(y_trues)):
            if y_trues[i] == 2.:
                y_trues_str[i] = 'Unknown'
            if y_trues[i] == 1.:
                y_trues_str[i] = 'Seborrheic Keratosis'
            if y_trues[i] == 0.:
                y_trues_str[i] = 'Melanoma'
                nearest_indices = []

        # pca reduction
        pca = PCA(n_components=10)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)

        for i in tqdm(range(len(embeddings))):
            if not nearest:
                nearest_indices.append(farthest_neighbors(embeddings[i],embeddings))
            else:
                nearest_indices.append(nearest_neighbors(embeddings[i],embeddings))
        nearest_indices = np.array(nearest_indices)

        # omega metric
        omega = 0
        for i in range(len(nearest_indices)):
            omega+= (y_trues[nearest_indices[i][0]] == y_trues[nearest_indices[i][1]])*1
        omega = omega / len(nearest_indices)
        print(f'Omega: {float(omega):.2f}')
        return

        for i in tqdm(range(min([len(nearest_indices),samples]))):
            neighbourhood = nearest_indices[i]
            base_dir = f"{directory}/{sub_dir}/{neighbourhood[0 if nearest else -1]}"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            ref_img = xs[neighbourhood[0 if nearest else -1]]
            cv2.imwrite(f'{base_dir}/Reference_{y_trues_str[neighbourhood[0 if nearest else -1]]}.png',cv2.cvtColor(ref_img.astype(np.uint8),cv2.COLOR_BGR2RGB))
            for neighbour in range(1 if nearest else 0,nbr_neighbors+1):
                n_index = neighbourhood[neighbour]
                cv2.imwrite(f'{base_dir}/{neighbour}_{y_trues_str[n_index]}.png',cv2.cvtColor(xs[n_index].astype(np.uint8),cv2.COLOR_BGR2RGB))
                plt.close()
        

# 0 = Melanoma
# 1 = seborrheic_keratosis
# 2 = Unknown




def get_nearest_neighbour_monuseg(num_classes,channels,directory,model,data, class_type='segmentation',nearest=False):
    nbr_neighbors=2
    samples = 80
    latent_dim = None
    sub_dir = 'nearest_neighbours'
    if not nearest:
        sub_dir = 'farthest_neighbours'
    #if num_classes == 1:
    if False:
        print('not implemented ...')
    else:
        Path(f"{directory}/{sub_dir}").mkdir(parents=True, exist_ok=True)
        

        composition = ModuleList()
        if class_type == 'resnet_classifier':
            composition.append(model.composition[0])
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            latent_dim = 2048
        if class_type == 'segmentation':
            composition.append(model.composition[0].cuda())
            composition.append(SegSelector())
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            
        latent_dim = 2048
        embeddings = np.zeros((0,latent_dim))
        y_trues = []
        xs = np.zeros((0,1000,1000,channels))
        for batch in tqdm(data.test_dataloader()):
            x,y_true = batch
            y_pred = x.to('cuda')
            for item in composition:
                y_pred = item(y_pred)
            y_pred = y_pred.squeeze()
            embeddings = np.concatenate((embeddings,y_pred.cpu().detach()),axis=0)
            xs = np.concatenate((xs,x.permute(0,2,3,1).cpu().detach()*255),axis=0)
            y_trues = y_trues + list(y_true)

        # pca reduction
        pca = PCA(n_components=10)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)


        nearest_indices = []
        for i in tqdm(range(len(embeddings))):
            if not nearest:
                nearest_indices.append(farthest_neighbors(embeddings[i],embeddings))
            else:
                nearest_indices.append(nearest_neighbors(embeddings[i],embeddings))
        nearest_indices = np.array(nearest_indices)


        # omega metric
        omega = 0
        for i in range(len(nearest_indices)):
            omega+= (y_trues[nearest_indices[i][0]] == y_trues[nearest_indices[i][1]])*1
        omega = omega / len(nearest_indices)
        print(f'Omega: {float(omega):.2f}')
        return


        for i in tqdm(range(min([len(nearest_indices),samples]))):
            neighbourhood = nearest_indices[i]
            base_dir = f"{directory}/{sub_dir}/{neighbourhood[0 if nearest else -1]}"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            ref_img = xs[neighbourhood[0 if nearest else -1]]
            cv2.imwrite(f'{base_dir}/Reference_{y_trues[neighbourhood[0 if nearest else -1]]}.png',cv2.cvtColor(ref_img.astype(np.uint8),cv2.COLOR_BGR2RGB)[256:768,256:768,:])
            for neighbour in range(1 if nearest else 0,nbr_neighbors+1):
                n_index = neighbourhood[neighbour]
                cv2.imwrite(f'{base_dir}/{neighbour}_{y_trues[n_index]}.png',cv2.cvtColor(xs[n_index].astype(np.uint8),cv2.COLOR_BGR2RGB)[256:768,256:768,:])
                plt.close()
        

# 0 = Melanoma
# 1 = seborrheic_keratosis
# 2 = Unknown