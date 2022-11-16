from pytorch_lightning import Callback, LightningModule, Trainer
import numpy as np
from DLIP.utils.metrics.inst_seg_metrics import inst_seg_metrics_dict
from DLIP.utils.metrics.inst_seg_metrics import remap_label
from DLIP.models.zoo.compositions.detectron_instance import get_mask_encoding
from typing import List

def calc_metric(name, gt_mask, pred_mask):
    score = inst_seg_metrics_dict[name](remap_label(gt_mask), remap_label(pred_mask))

class DetectronLogInstSegMetricsCallback(Callback):
    """
    Logs instance segmentation metrics
    """
    def __init__(
        self,
        ):
        super().__init__()
        self.inst_seg_metrics_dict = inst_seg_metrics_dict
        self.result_metrics = dict()

        for keys in self.inst_seg_metrics_dict.keys():
            self.result_metrics[keys] = list()

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        # do prediction
        #pl_module.eval()
        for batch in trainer.datamodule.test_dataloader():
            if not isinstance(batch, List):
                batch = [batch]
            
            #prediction = pl_module.model([elem.to(pl_module.device) for elem in batch]) :TODO GPU support
            prediction = pl_module.model(batch)
            for i_b in range(len(prediction)):
                gt_mask = get_mask_encoding(batch[i_b]["instances"].gt_masks.tensor)
                pred_mask = get_mask_encoding(prediction[i_b]["instances"].pred_masks)
                for key,value in self.inst_seg_metrics_dict.items():
                    self.result_metrics[key].append(value(remap_label(gt_mask), remap_label(pred_mask)))

        for key,value in self.inst_seg_metrics_dict.items():
            trainer.logger.log_metrics({
                key: 
                np.mean(self.result_metrics[key])
            })
