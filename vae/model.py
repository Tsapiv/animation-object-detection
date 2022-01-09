import urllib.parse
from abc import ABC

import torch
from pl_bolts.models import VAE
from pytorch_lightning import LightningModule
from torch import nn

from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import resnet18_encoder
from triplets import mAP


class WrappedVAE(VAE, ABC):

    def __init__(self, input_height: int = 32, nth: int = 100, **kwargs):
        super().__init__(input_height, **kwargs)
        self.nth = nth

    def encode(self, x):
        x = self.encoder(x)
        return x

    def step(self, batch, batch_idx):
        loss, logs = super().step(batch, batch_idx)
        if batch_idx % self.nth == 0:
            verbose = False
            if self.training:
                verbose = True
                self.eval()
            data, target = batch
            descriptors = self.encode(data)
            mapk = mAP(descriptors, target)
            logs = {**logs, **{"mapk": mapk}}
            if verbose:
                self.train()
        return loss, logs
