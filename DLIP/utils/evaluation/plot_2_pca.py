from pathlib import Path
from tkinter import N
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='lower right')


def plot_2_pca(num_classes,directory,model,data, class_type='resnet_classifier'):
    latent_dim = None
    if num_classes == 1:
        print('not implemented ...')
    else:
        Path(f"{directory}/2_pca").mkdir(parents=True, exist_ok=True)
        
        composition = ModuleList()
        if class_type == 'resnet_classifier':
            composition.append(model.composition[0])
            composition.append(nn.AdaptiveAvgPool2d((1,1)))
            latent_dim = 2048

        embeddings = np.zeros((0,latent_dim))
        y_trues = np.zeros((0,num_classes))
        xs = np.zeros((0,256,256,3))
        for batch in tqdm(data.test_dataloader()):
            x,y_true = batch
            y_pred = x
            for item in composition:
                item = item.to('cuda')
                y_pred = item(y_pred.to('cuda'))
            y_pred = y_pred.squeeze()
            embeddings = np.concatenate((embeddings,y_pred.cpu().detach()),axis=0)
            y_trues = np.concatenate((y_trues,y_true.detach().cpu()),axis=0)
            xs = np.concatenate((xs,x.permute(0,2,3,1).cpu().detach()*255),axis=0)
        
        pca = PCA(n_components=2).fit(embeddings)
        variance_ratio = pca.explained_variance_ratio_
        samples_transformed = pca.transform(embeddings)
        colors = [(random.randint(1,255), random.randint(1,255), random.randint(1,255)) for _ in range(y_trues.shape[1])]
        for i in range(len(samples_transformed)):
            sample_transformed = samples_transformed[i]
            color = colors[np.argmax(y_trues[i])]
            scatter = plt.scatter(
                x=sample_transformed[0],
                y=sample_transformed[1],
                color=np.array(list(color))/255,
                label=np.argmax(y_trues[i])
            )
        legend_without_duplicate_labels(plt)
        plt.title(f'Variance Ratio: {variance_ratio}')
        plt.savefig(f"{directory}/2_pca/pca.png")
        plt.close()
        
        
            
        