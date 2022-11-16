from pytorch_lightning import Callback, LightningModule, Trainer
import copy
import logging
import torch
from DLIP.utils.evaluation.cka_simplified import CKASimplified


class LogCKACallback(Callback):
    
    def __init__(self,benchmark_model_path):
        super().__init__()
        self.benchmark_model_path = benchmark_model_path
        
        
    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        if self.benchmark_model_path is None:
            ref_model = pl_module
            logging.info('No ref model for CKA calculation, taking the model itself.')
        else:
            ref_model = copy.deepcopy(pl_module)
            missing_keys,unmatched_keys = ref_model.load_state_dict(torch.load(self.benchmark_model_path)['state_dict'])
            logging.info('Loaded Reference model for CKA calculation.')
            logging.info(f'Missing Keys: {missing_keys}.')
            logging.info(f'Unmatched Keys: {unmatched_keys}.')
        cka = CKASimplified(pl_module, ref_model,
                model1_name="Trained Model",
                model2_name="Ref Model",
                device='cuda')
        cka_value = cka.compare(trainer.datamodule.test_dataloader())
        pl_module.log("CKA_to_ref_model", cka_value, prog_bar=True)
        
        
        
        