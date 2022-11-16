from pytorch_lightning import Callback, LightningModule, Trainer
import numpy as np
from DLIP.utils.metrics.inst_seg_metrics import inst_seg_metrics_dict
from DLIP.utils.metrics.inst_seg_metrics import remap_label
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor

class LogInstSegMetricsCallback(Callback):
    """
    Logs instance segmentation metrics
    """
    def __init__(
        self,
        inst_seg_pp_params
        ):
        super().__init__()
        self.inst_seg_metrics_dict = inst_seg_metrics_dict
        self.dist_map_post_processor = DistMapPostProcessor(**inst_seg_pp_params)
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
        trainer.datamodule.test_dataset.label_raw_mode = True
        for batch in trainer.datamodule.test_dataloader():
            x, y_true = batch
            y_true = y_true.permute(0, 3, 1, 2)
            y_pred = pl_module(x.to(pl_module.device))
            y_pred = y_pred.detach().cpu().numpy()
            for i_x in range(y_pred.shape[0]):
                y_inst = self.dist_map_post_processor.process(y_pred[i_x,0,:], x[i_x,:])
                for key,value in self.inst_seg_metrics_dict.items():
                    self.result_metrics[key].append(value(remap_label(y_true[i_x,0,:].cpu().numpy()), remap_label(y_inst)))

        for key,value in self.inst_seg_metrics_dict.items():
            trainer.logger.experiment.add_scalar(
                key, 
                np.mean(self.result_metrics[key]), 
                global_step=trainer.global_step
            )

        trainer.datamodule.test_dataset.label_raw_mode = False