from cv2 import COLOR_BGR2RGB
import matplotlib
matplotlib.use('Agg')

import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch
from tqdm import tqdm

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
from pytorch_grad_cam import GradCAM
from pathlib import Path


class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """
    def __init__(self, mask):
        self.mask = mask.cuda()

    def __call__(self, model_output):
        return (model_output * self.mask).sum()

def calculate_cams(model,data,directory):

    j=0
    for batch in tqdm(data.test_dataloader()):
        x,y_true = batch
        y_pred = model(x.cuda())

        #target_layers = [model.composition[0].backbone[4][0]] #resnet
        target_layers = [model.composition[0].backbone[4]] # unet
        
        targets = [SemanticSegmentationTarget(y_true[0].permute(2,0,1))]

        Path(f"{directory}/EigenCAM").mkdir(parents=True, exist_ok=True)
        Path(f"{directory}/FullGrad").mkdir(parents=True, exist_ok=True)
        Path(f"{directory}/XGradCAM").mkdir(parents=True, exist_ok=True)

        for i in range(len(x)):
            if j == 10 or j == 11 or j == 13:
                targets = [SemanticSegmentationTarget(y_true[i].permute(2,0,1))]
                with EigenCAM(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
                    grayscale_cam = cam(input_tensor=x[i:i+1],targets=targets)[0, :]
                    cam_image = show_cam_on_image((x[i].permute(1,2,0)).numpy(), grayscale_cam, use_rgb=True)
                    cv2.imwrite(f'{directory}/EigenCAM/{j}.png',cam_image)
                    cv2.imwrite(f'{directory}/EigenCAM/{j}_pred_mask.png',((y_pred[i]>0.5).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
                    cv2.imwrite(f'{directory}/EigenCAM/{j}_true_mask.png',(y_true[i].detach().cpu().numpy()*255).astype(np.uint8))
                with FullGrad(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
                    grayscale_cam = cam(input_tensor=x[i:i+1],targets=targets)[0, :]
                    cam_image = show_cam_on_image((x[i].permute(1,2,0)).numpy(), grayscale_cam, use_rgb=True)
                    cv2.imwrite(f'{directory}/FullGrad/{j}.png',cam_image)
                    cv2.imwrite(f'{directory}/FullGrad/{j}_pred_mask.png',((y_pred[i]>0.5).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
                    cv2.imwrite(f'{directory}/FullGrad/{j}_true_mask.png',(y_true[i].detach().cpu().numpy()*255).astype(np.uint8))
                with XGradCAM(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
                    grayscale_cam = cam(input_tensor=x[i:i+1],targets=targets)[0, :]
                    cam_image = show_cam_on_image((x[i].permute(1,2,0)).numpy(), grayscale_cam, use_rgb=True)
                    cv2.imwrite(f'{directory}/XGradCAM/{j}.png',cam_image)
                    cv2.imwrite(f'{directory}/XGradCAM/{j}_pred_mask.png',((y_pred[i]>0.5).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
                    cv2.imwrite(f'{directory}/XGradCAM/{j}_true_mask.png',(y_true[i].detach().cpu().numpy()*255).astype(np.uint8))
            j+=1
            if j>13:
                break
        if j>13:
                break






def calculate_accuracies(num_classes,directory,model,data,channels):
    Path(f"{directory}/missclassifies").mkdir(parents=True, exist_ok=True)
    Path(f"{directory}/correct").mkdir(parents=True, exist_ok=True)
    y_preds = np.zeros((0,num_classes))
    y_trues = np.zeros((0,num_classes))
    xs = np.zeros((0,256,256,channels))

    for batch in tqdm(data.test_dataloader()):
        x,y_true = batch
        y_pred = model(x)
        y_preds = np.concatenate((y_preds,y_pred.detach().cpu()),axis=0)
        y_trues = np.concatenate((y_trues,y_true.detach().cpu()),axis=0)
        xs = np.concatenate((xs,x.permute(0,2,3,1).cpu().detach()*255),axis=0)

    y_preds = y_preds.squeeze()
    y_trues = y_trues.squeeze()
    
    if num_classes == 1:
        y_preds_binary = ((y_preds > 0.5) * 1.).squeeze()

        accuracy = np.sum(y_preds_binary.squeeze() == y_trues) / len(y_trues.squeeze())

        # accuracy 1
        idxs = np.argwhere(y_trues == 1.)
        accuracy_1 = np.sum(y_preds_binary[idxs].squeeze() == y_trues[idxs].squeeze()) / len(y_trues[idxs].squeeze())

        # accuracy 0
        idxs = np.argwhere(y_trues == 0.)
        accuracy_0 = np.sum(y_preds_binary[idxs].squeeze() == y_trues[idxs].squeeze()) / len(y_trues[idxs].squeeze())

        with open(f'{directory}/stats.txt', 'w') as file:
            file.write(f'Accuracy: {accuracy}\n')
            file.write(f'Accuracy 1: {accuracy_1}\n')
            file.write(f'Accuracy 0: {accuracy_0}\n')

        cam = GradCAM(model=model, target_layers=[model.composition[0].backbone[-1][-1]], use_cuda=True)

        # all misclassifies
        miss_indx = np.argwhere((y_preds_binary != y_trues).squeeze()).squeeze()
        for idx in tqdm(miss_indx):
            x = xs[idx]
            true = y_trues[idx]
            pred = y_preds_binary[idx]
            plt.imshow(x.astype(np.uint8))
            plt.title(f'True: {true}, pred: {pred}')
            plt.savefig(f'{directory}/missclassifies/{idx}.png')
            plt.close()
            x = torch.Tensor(x/255.).permute(2,0,1)
            grayscale_cam = cam(input_tensor=x.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(x.permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=False)
            cv2.imwrite(f'{directory}/missclassifies/{idx}_cam.png',visualization)
            
        # some corrects
        miss_indx = np.argwhere((y_preds_binary == y_trues).squeeze()).squeeze()
        for idx in tqdm(miss_indx[:50]):
            x = xs[idx]
            true = y_trues[idx]
            pred = y_preds_binary[idx]
            plt.imshow(x.astype(np.uint8))
            plt.title(f'True: {true}, pred: {pred}')
            plt.savefig(f'{directory}/correct/{idx}.png')
            plt.close()
            x = torch.Tensor(x/255.).permute(2,0,1)
            grayscale_cam = cam(input_tensor=x.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(x.permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=False)
            cv2.imwrite(f'{directory}/correct/{idx}_cam.png',visualization)
    else:
        y_trues_max = np.argmax(y_trues,axis=1)
        y_preds_max = np.argmax(y_preds,axis=1)
        overall_acc = np.sum(y_preds_max == y_trues_max) / len(y_trues_max)
        with open(f'{directory}/stats.txt', 'a') as file:
            file.write(f'Accuracy: {overall_acc}\n')
        cam = GradCAM(model=model, target_layers=[model.composition[0].backbone[-1][-1]], use_cuda=True)
        for clazz in range(num_classes):
            true_class_indices = np.argwhere(y_trues_max == clazz).squeeze()
            t = y_trues_max[true_class_indices]
            p = y_preds_max[true_class_indices]
            accuracy = np.sum(t == p) / len(t)
            with open(f'{directory}/stats.txt', 'a') as file:
                file.write(f'Accuracy {clazz}: {accuracy}\n')
            Path(f"{directory}/missclassifies/{clazz}").mkdir(parents=True, exist_ok=True)
            Path(f"{directory}/correct/{clazz}").mkdir(parents=True, exist_ok=True)
            
        # all misclassifies
        miss_indx = np.argwhere(y_preds_max != y_trues_max).squeeze()
        for idx in tqdm(miss_indx):
            x = xs[idx]
            true = np.argmax(y_trues[idx])
            pred = np.argmax(y_preds[idx])
            plt.imshow(x.astype(np.uint8))
            plt.title(f'True: {true}, pred: {pred}')
            plt.savefig(f'{directory}/missclassifies/{true}/{idx}.png')
            plt.close()
            x = torch.Tensor(x/255.).permute(2,0,1)
            grayscale_cam = cam(input_tensor=x.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(x.permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=False)
            cv2.imwrite(f'{directory}/missclassifies//{true}/{idx}_cam.png',visualization)
            
        # some corrects
        miss_indx = np.argwhere(y_preds_max == y_trues_max).squeeze()
        for idx in tqdm(miss_indx[:50]):
            x = xs[idx]
            true = np.argmax(y_trues[idx])
            pred = np.argmax(y_preds[idx])
            plt.imshow(x.astype(np.uint8))
            plt.title(f'True: {true}, pred: {pred}')
            plt.savefig(f'{directory}/correct/{true}/{idx}.png')
            plt.close()
            x = torch.Tensor(x/255.).permute(2,0,1)
            grayscale_cam = cam(input_tensor=x.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(x.permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=False)
            cv2.imwrite(f'{directory}/correct/{true}/{idx}_cam.png',visualization)