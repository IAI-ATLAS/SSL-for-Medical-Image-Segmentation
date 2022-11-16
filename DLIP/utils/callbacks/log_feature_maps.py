from typing import Sequence
from pytorch_lightning import Callback, LightningModule, Trainer
import time
import wandb
from tqdm import tqdm

class LogFeatureMaps(Callback):
    def __init__(
        self
        ):
        super().__init__()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx,
        dataloader_idx
    ):
        if batch_idx == 0:
            features = pl_module.composition[0](batch[0])
            features = [features[0]] + features[1]
            for layer in range(len(features)): 
                images = [wandb.Image(x.detach().cpu().numpy()) for x in features[layer][0][:30]]
                wandb.log({f"feature maps {layer}": images})

        
