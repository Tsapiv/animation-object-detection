import urllib.parse

import numpy as np
import torch
from pl_bolts.models import VAE
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn

from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import resnet18_encoder
from utils import mAP


class ResNet18Encoder(LightningModule):
    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"),
        "stl10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"),
    }

    def __init__(
            self,
            first_conv: bool = False,
            maxpool1: bool = False,
            nth: int = 100,
            lr: float = 1e-4,
            **kwargs,
    ):
        """
        Args:
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            lr: learning rate for Adam
        """

        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.nth = nth
        self.__criterion = nn.TripletMarginLoss()
        self.encoder = resnet18_encoder(first_conv, maxpool1)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def step(self, batch, batch_idx, phase):
        data, target = batch
        output = [self(sample) for sample in data]
        loss = self.__criterion(*output)
        if batch_idx % self.nth == 0:
            combined_data = torch.cat(output[:-1])
            combined_target = torch.cat([target, target])
            mapk = mAP(combined_data, combined_target)
            self.log_dict({f"{phase}_mapk": mapk}, on_step=True, on_epoch=False)
        self.log_dict({f"{phase}_loss": loss}, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_dict({f"val_loss": np.mean(outputs)}, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
