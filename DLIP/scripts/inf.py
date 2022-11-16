import cv2
import logging
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.merge_configs import merge_configs

from DLIP.utils.loading.split_parameters import split_parameters
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from skimage.transform import resize

def get_mask_encoding(tensor):
    mask = np.zeros((tensor.shape[1:]), dtype=np.int16)
    for i_i in range(tensor.shape[0]):
        mask[tensor[i_i].detach().cpu().numpy()] =  i_i + 1
    return mask

logging.basicConfig(level=logging.INFO)

config_files    = "/home/ws/sc1357/projects/devel/src/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/base_cfg/cfg_inst_seg_base.yaml"
ckpt_file       = "/home/ws/sc1357/data/0556/dnn_weights.ckpt"

cfg_yaml = merge_configs(config_files)
experiment_name=cfg_yaml['experiment.name']['value']

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=None,
    config_name=None,
    disabled=True
)

seed_everything(seed=config['experiment.seed'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

data = load_data_module(parameters_splitted["data"], do_val_init=False)
model = load_model(parameters_splitted["model"], checkpoint_path_str=ckpt_file)

metric = list()

model.model.training = False
model.model.eval()

for sample in data.test_dataset:
    input = [sample]
    resized_img = F.resize(sample["image"], (4000,4000)).to("cpu")
    input[0]["image"] = resized_img

    res = model.model(input)

    pred_mask = get_mask_encoding(res[0]["instances"].pred_masks)
    gt_mask = get_mask_encoding(input[0]["instances"].gt_masks.tensor)

    metric.append(get_fast_aji_plus(
        remap_label(gt_mask),
        remap_label(cv2.resize(pred_mask,(1000,1000),interpolation=cv2.INTER_NEAREST))
    ))

print(f"Final score: {np.mean(metric)}")