from pl_bolts.models import AE
from torch import nn


class ResNet18Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ae = AE(*args, **kwargs)
        ae = ae.from_pretrained('cifar10-resnet18')
        self.__model = ae.encoder

    def forward(self, x):
        return self.__model(x)
