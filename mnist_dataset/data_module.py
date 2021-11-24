from abc import ABC
from typing import Callable
from torchvision import transforms
from pl_bolts.datamodules import MNISTDataModule
from utils import CentralPad, channel_expander


class EMNISTDataModule(MNISTDataModule, ABC):
    def default_transforms(self) -> Callable:
        return transforms.Compose(
            [super().default_transforms(), CentralPad((32, 32)), transforms.Lambda(channel_expander)])
