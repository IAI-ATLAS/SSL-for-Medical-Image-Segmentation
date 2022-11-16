from pytorch_lightning import Callback, LightningModule, Trainer
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor
from skimage.color import label2rgb
import wandb
import cv2
from DLIP.utils.visualization.inst_seg_contour import visualize_instances_map


class ImageLogInstSegCallback(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(
        self,
        inst_seg_pp_params,
        num_img_log=6
        ):
        """
        Args:
        """
        super().__init__()
        self.dist_map_post_processor = DistMapPostProcessor(**inst_seg_pp_params)
        self.num_img_log = num_img_log
        


    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        # do prediction
        #pl_module.eval()
        prediction_lst_ol = list()
        prediction_lst_ct = list()
        gt_lst_ol         = list()
        gt_lst_ct         = list()
        trainer.datamodule.test_dataset.label_raw_mode = True

        for i_x in range(self.num_img_log):
            x, y_true = trainer.datamodule.test_dataset[i_x]
            y_pred = pl_module(x.unsqueeze(0).to(pl_module.device))
            y_pred = y_pred.detach().cpu().numpy()
            y_inst_pred = self.dist_map_post_processor.process(y_pred[0,:], x)[0,:]
            trainer.datamodule.test_dataset.set_raw_mode(True)
            x_raw, _ = trainer.datamodule.test_dataset[i_x]
            trainer.datamodule.test_dataset.set_raw_mode(False)
            y_inst_pred_rgb = label2rgb(y_inst_pred, bg_label=0, image=x_raw, image_alpha=0.5)
            y_inst_gt_rgb   = label2rgb(y_true.squeeze().cpu().numpy(), bg_label=0, image=x_raw, image_alpha=0.5)
            prediction_lst_ol.append(wandb.Image(y_inst_pred_rgb.copy(), caption=f"{i_x+1}") )
            prediction_lst_ct.append(wandb.Image(visualize_instances_map(cv2.cvtColor(x_raw,cv2.COLOR_GRAY2BGR),y_inst_pred), caption=f"{i_x+1}") )
            gt_lst_ol.append(wandb.Image(y_inst_gt_rgb.copy(), caption=f"{i_x+1}") )
            gt_lst_ct.append(wandb.Image(visualize_instances_map(cv2.cvtColor(x_raw,cv2.COLOR_GRAY2BGR),y_true.squeeze().cpu().numpy()), caption=f"{i_x+1}"))

        wandb.log({f"test/pred_overlay": prediction_lst_ol})
        wandb.log({f"test/pred_contour": prediction_lst_ct})
        wandb.log({f"test/gt_overlay": gt_lst_ol})
        wandb.log({f"test/gt_contour": gt_lst_ct})

        trainer.datamodule.test_dataset.label_raw_mode = False