from typing import Sequence
from pytorch_lightning import Callback, LightningModule, Trainer
import time
import logging

class IncreaseSSLImageSizeCallback(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(self,increase_factor=2):
        """
        Args:
        """
        super().__init__()
        self.increase_factor = increase_factor
        self.frequency = 2000
        self.max_size = 512

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
    ):
        if (pl_module.current_epoch % self.frequency) == 0 and pl_module.current_epoch != 0:
            logging.info(f'Increasing Image Size by factor {self.increase_factor}')
            old_size = trainer.datamodule.train_dataloader().dataset.transforms[0].transform['pre'][0].height
            new_size = old_size * self.increase_factor
            if new_size >= self.max_size:
                logging.info('Max Image Size Reached, ignoring.')
                return
            logging.info(f'Increasing Image Size from {old_size} to {new_size}')
            
            trainer.datamodule.train_dataloader().dataset.edit_sample_size(new_size,new_size)
            trainer.datamodule.val_dataloader().dataset.edit_sample_size(new_size,new_size)
            trainer.datamodule.test_dataloader().dataset.edit_sample_size(new_size,new_size)