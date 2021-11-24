import os
from abc import ABC
from os import listdir
from typing import Optional, Callable

import numpy as np
from torch.utils.data import Dataset


class GoogleDoodleDataset(Dataset, ABC):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            *args, **kwargs
    ) -> None:
        super(GoogleDoodleDataset, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = 0
        self.data, self.targets = self._load_data()

    def _load_data(self):
        data = [np.load(os.path.join(self.root, f)) for f in listdir(self.root) if f.endswith('.npy')]
        targets = [np.full(len(category), idx) for idx, category in enumerate(data)]
        self.classes = len(data)
        return np.reshape(np.vstack(data), (-1, 28, 28)), np.hstack(targets)

    def num_classes(self):
        return self.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
