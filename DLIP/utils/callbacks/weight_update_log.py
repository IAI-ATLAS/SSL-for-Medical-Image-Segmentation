from typing import Sequence
from pytorch_lightning import Callback, LightningModule, Trainer
import time
from tqdm import tqdm

class WeightUpdateLog(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(
        self
        ):
        super().__init__()
        self.encoder_weights = None
        self.decoder_weights = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx,
        dataloader_idx
    ):
        encoder_weights_sum = 0
        enc = list(pl_module.composition[0].parameters())
        for i in range(len(enc)):
            #if i%20 == 0:
            encoder_weights_sum+=float(enc[i].abs().sum())
        decoder_weights_sum = 0
        dec = list(pl_module.composition[1].parameters())
        for i in  range(len(dec)):
            #if i%20 == 0:
            decoder_weights_sum+=float(dec[i].abs().sum())
        if self.encoder_weights is None:
            self.encoder_weights = encoder_weights_sum
            self.decoder_weights = decoder_weights_sum
        if type(self.encoder_weights) is list:
            self.encoder_weights.append(encoder_weights_sum)
            self.decoder_weights.append(decoder_weights_sum)
            pl_module.log("train/enc_diff", abs(self.encoder_weights[-1]-self.encoder_weights[-2]))
            pl_module.log("train/dec_diff", abs(self.decoder_weights[-1]-self.decoder_weights[-2]))
        else:
            self.encoder_weights = [abs(encoder_weights_sum-self.encoder_weights)]
            self.decoder_weights = [abs(decoder_weights_sum-self.decoder_weights)]
