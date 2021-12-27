import numpy as np
from torch.utils.data import Dataset


class TripletsDatasetWrapper(Dataset):
    """
    Load the MNIST dataset in pairs of similar(positive)/non-similar(negative) pairs
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.transform = self.dataset.transform
        self.labels = self.dataset.targets
        self.data = self.dataset.data
        self.class_idx = [(self.labels == x).nonzero()[0] for x in range(self.dataset.num_classes())]
        # indices of images for each class

    def __getitem__(self, index):
        # anchor image
        anchor, label = self.dataset[index]
        # draw another positive (1) or negative (0) image

        # choose an image with the same label as the anchor - avoid itself
        pos_index = index
        while pos_index == index:
            pos_index = np.random.choice(self.class_idx[label])
        positive, _ = self.dataset[pos_index]

        negative, _ = self.dataset[
            np.random.choice(self.class_idx[np.random.choice(np.setdiff1d(range(self.dataset.num_classes()), label))])]
        return [anchor, positive, negative], label

    def __len__(self):
        return len(self.data)
