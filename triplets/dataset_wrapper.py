from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TripletsDatasetWrapper(Dataset):
    """
    Load the MNIST dataset in pairs of similar(positive)/non-similar(negative) pairs
    """

    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset: Dataset = dataset
        self.train: bool = kwargs['train']
        self.transform = self.dataset.transform
        self.labels: torch.Tensor = self.dataset.targets
        self.data: torch.Tensor = self.dataset.data
        self.triplets = None
        self.triplets = self.generate_triplets(self.labels)
        # indices of images for each class

    @staticmethod
    def generate_triplets(labels):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels)
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        # full_idx = set(unique_labels)
        # already_idxs = set()

        for _ in tqdm(range(len(labels))):
            c1 = np.random.randint(0, n_classes)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 1:
                continue
            elif len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        # anchor image
        anchor, positive, negative = self.triplets[index]
        anchor, label = self.dataset[anchor]
        if self.train:
            return (anchor, self.dataset[positive][0], self.dataset[negative][0]), label
        return (anchor, self.dataset[positive][0]), label

    def __len__(self):
        return len(self.triplets)
