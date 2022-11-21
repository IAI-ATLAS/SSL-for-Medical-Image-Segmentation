from cv2 import COLOR_BGR2RGB
import matplotlib

from DLIP.utils.evaluation.accuracy_with_dirs import calculate_accuracies, calculate_cams
from DLIP.utils.evaluation.calculate_cka import calculate_cka
from DLIP.utils.evaluation.cka_simplified import CKASimplified
from DLIP.utils.evaluation.nearest_neighbour_retrival import get_nearest_neighbour,get_nearest_neighbour_monuseg
from DLIP.utils.evaluation.plot_2_pca import plot_2_pca
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

directory = 'monuseg_cam/detco'

# autoencoder. derma?
#checkpoint_path = '/home/ws/kg2371/results/first-shot/IsicDermoDataModule/UnetSemantic/0001/dnn_weights.ckpt'   

#checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/BaseInstanceSegmentationDataModule/UnetInstance/0083/dnn_weights.ckpt'

#monuseg ref
#ref_checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/BaseInstanceSegmentationDataModule/UnetInstance/0117/dnn_weights.ckpt'

#derma ref -> Full UNet / Fuer autoencoder
#ref_checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/IsicDermoDataModule/UnetSemantic/0139/dnn_weights.ckpt'



# /lsdf/kit/iai/projects/iai-aida/Daten_Schilling/2022_07_15_DAL_AE/first-shot/BaseInstanceSegmentationDataModule/UnetAE/0001/dnn_weights.ckpt

# derma ref
#ref_checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/IsicDermoDataModule/UnetSemantic/0150/dnn_weights.ckpt'

checkpoint_path = '/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/results/first-shot/IsicDermoDataModule/UnetSemantic/0101/dnn_weights.ckpt'

print('CHECKPOINT PATH')
print(checkpoint_path)

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

args = parse_arguments()
config_files, result_dir = args["config_files"], args["result_dir"]

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

# set wandb to disabledresults/first-shot/IsicDermoDataModule/UnetSemantic/0106
cfg_yaml['wandb.mode'] = {'value' : 'disabled'}
# Encoder should not be frozen for evaluation
if 'model.params.encoder_frozen' in cfg_yaml:
    cfg_yaml['model.params.encoder_frozen'] = {'value' : False}

experiment_dir, config_name = prepare_directory_structure(
    base_path=base_path,
    experiment_name=experiment_name,
    data_module_name=cfg_yaml['data.datamodule.name']['value'],
    model_name=cfg_yaml['model.name']['value']
)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=experiment_dir,
    config_name=config_name
)
logging.warn(f"Working Dir: {os.getcwd()}")
seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], 
    checkpoint_path_str=checkpoint_path              
)
# ref_model  = load_model(parameters_splitted["model"], 
#     checkpoint_path_str=ref_checkpoint_path                 
# )

data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(train_params=parameters_splitted['train'], result_dir=experiment_dir, run_name=wandb.run.name, data=data,config=config)

model = model.cuda()

# print('CALCULATING CAMs')
# calculate_cams(
#     model=model,
#     data=data,
#     directory=directory,
# )
# exit()


# print('CALCULATING ACCURACIES')
# calculate_accuracies(
#     num_classes=1,
#     channels=1,
#     directory=directory,
#     model=model,
#     data=data,
# )

# print('CALCULATING NEAREST NEIGHBOURS')
get_nearest_neighbour_monuseg(
    num_classes=1,
    channels=3,
    directory=directory,
    model=model,
    data=data,
    nearest=True
)
exit()
# get_nearest_neighbour_monuseg(
#     num_classes=1,
#     channels=3,
#     directory=directory,
#     model=model,
#     data=data,
#     nearest=False
# )
# exit()

# print('CALCULATING 2 PCA')
# plot_2_pca(
#     num_classes=1,
#     directory=directory,
#     model=model,
#     data=data,
# )

# calculate_cka(
#     data=data,
#     directory=directory,
#     model=model,
#     ref_model=ref_model
# )
# exit()


from torch_cka import CKA
from torchvision.models import resnet18

model1 = resnet18(pretrained=True)  # Or any neural network of your choice
model2 = resnet18(pretrained=True)