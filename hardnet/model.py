from abc import ABC

import torch
from pl_bolts.models.autoencoders import resnet18_encoder
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn

from hardnet.loss import HardNetLoss, CorrelationPenaltyLoss
from utils import mAP


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


def input_norm(x):
    flat = x.view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(
        -1).unsqueeze(1).expand_as(x)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass


class HardNet(LightningModule, ABC):
    """HardNet model definition
    """

    def __init__(self,
                 first_conv: bool = False,
                 maxpool1: bool = False,
                 nth: int = 2,
                 lr: float = 1e-2,
                 **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        self.corr_penalty = CorrelationPenaltyLoss()
        self.l2norm = L2Norm()
        self.lr = lr
        self.nth = nth
        self.size = 512
        self.labels, self.distances = [], []
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=(8, 8), bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        # self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.encoder.apply(weights_init)
        return

    def step(self, batch, batch_idx, phase):
        rez = {}
        (a, p, n), label = batch
        out_a, out_p = self(a), self(p)
        loss = HardNetLoss(out_a, out_p)
        if phase == 'train':
            loss += self.corr_penalty(out_a)
        if batch_idx % self.nth == 0 and phase == 'val':
            mapk = mAP(out_a[:self.size], label[:self.size])
            rez['mapk'] = mapk
            self.log_dict({f"step_{phase}_mapk": mapk}, on_step=True, on_epoch=False)
        # if phase == 'val':
        #     dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        #     self.distances.append(dists.data.cpu().numpy().reshape(-1, 1))
        #     ll = label.data.cpu().numpy().reshape(-1, 1)
        #     self.labels.append(ll)
        rez['loss'] = loss
        self.log_dict({f"step_{phase}_loss": loss}, on_step=True, on_epoch=False)
        return rez

    def forward(self, input):
        x_features = self.encoder(input)
        x = x_features.view(x_features.size(0), -1)
        return x#self.l2norm(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_dict({f"train_loss": torch.mean(torch.Tensor([out['loss'] for out in outputs]))}, on_step=False,
                      on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # labels = np.vstack(self.labels)
        # distances = np.vstack(self.distances)
        # fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        # print(outputs)
        # self.log_dict({f"val_fpr95": fpr95}, on_step=False, on_epoch=True)
        self.log_dict({f"val_loss": torch.mean(torch.Tensor([out['loss'] for out in outputs]))}, on_step=False,
                      on_epoch=True)
        self.log_dict({f"val_mapk": torch.mean(torch.Tensor([out['mapk'] for out in outputs if 'mapk' in out]))}, on_step=False,
                      on_epoch=True)
        # self.labels, self.distances = [], []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
