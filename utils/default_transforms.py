from typing import Callable

from torchvision.transforms import Compose, ToTensor, Lambda, Normalize

from utils import CentralPad, channel_expander


def default_transform(normalize: bool = True) -> Callable:
    base_transforms = Compose(
        [ToTensor(), CentralPad((32, 32)), Lambda(channel_expander)])
    transforms = Compose(
        [base_transforms, Normalize(mean=(0.5,), std=(0.5,))]) if normalize else base_transforms
    return transforms
