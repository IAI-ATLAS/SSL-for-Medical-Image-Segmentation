from typing import Optional, Union
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
import torch

class AdvancedModelCheckpoint(ModelCheckpoint):
    
    def __init__(
        self,
        config,
        dirpath  = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        period: Optional[int] = None,
        every_n_val_epochs: Optional[int] = None
    ):
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end, period, every_n_val_epochs)
        self.config = dict(config)
    
    def _save_model(self, trainer: "pl.Trainer", filepath: str):
        
        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        trainer.save_checkpoint(filepath, self.save_weights_only)
        
        f = torch.load(filepath)
        f['config'] = self.config
        torch.save(f,filepath)