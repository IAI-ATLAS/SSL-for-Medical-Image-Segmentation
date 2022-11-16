import matplotlib
matplotlib.use('Agg')

import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure

from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

args = parse_arguments()
config_files, result_dir = args["config_files"], args["result_dir"]

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

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

seed_everything(seed=config['experiment.seed'])

parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"])
data = load_data_module(parameters_splitted["data"])

trainer = load_trainer(train_params=parameters_splitted['train'], result_dir=experiment_dir, run_name=wandb.run.name, data=data,config=config)

if 'train.cross_validation.n_splits' in cfg_yaml:
    cv_trainer = CVTrainer(
        trainer=trainer,
        n_splits=cfg_yaml['train.cross_validation.n_splits']['value']
    )
    cv_trainer.fit(model=model,datamodule=data)
else:
    trainer.fit(model, data)
    test_results = trainer.test(model=model, dataloaders=data.test_dataloader())
wandb.finish()
