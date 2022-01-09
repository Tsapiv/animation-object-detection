import os
from abc import ABC
from os import listdir
from typing import Optional, Callable
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset


class GoogleDoodleDataset(Dataset, ABC):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            classes: int = 10,
            ratio: float = 0.9999,
            *args, **kwargs
    ) -> None:
        super(GoogleDoodleDataset, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.ratio = ratio
        self.data, self.targets = self._load_data()

    def _load_data(self):
        filenames = [os.path.join(self.root, f) for f in listdir(self.root) if
                     f.endswith('.npy') and ('train' if self.train else 'test') in f]
        if len(filenames) != 2:
            raise RuntimeError('Dataset root directory (lacks of) / (contains redundant) files')
        data = list(map(np.load, filenames))
        _, data0, _, data1 = train_test_split(*data, test_size=self.ratio)
        return (data1, data0) if 'labels' in filenames[0] else (data0, data1)

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
