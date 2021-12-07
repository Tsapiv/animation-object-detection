import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
from google_doodle_dataset import GoogleDoodleDataset, GoogleDoodleDataModule

from metrics import distance_validation


class ModelSaver:
    def __init__(self, root, topk: int = 3, minimize: bool = True):
        self.topk = topk
        self.root = root
        self.minimize = minimize
        self.rank = []
        os.makedirs(self.root, exist_ok=True)

    def update(self, model, criterion, epoch):
        if len(self.rank) < self.topk:
            self.rank.append((criterion, self.generate_name(criterion, epoch)))
            torch.save(model, self.generate_name(criterion, epoch))
        else:
            self.rank.sort(key=lambda x: x[0] if self.minimize else -x[0])
            if self.rank[-1][0] > criterion:
                _, path = self.rank.pop()
                os.unlink(path)
                self.rank.append((criterion, self.generate_name(criterion, epoch)))
                torch.save(model, self.generate_name(criterion, epoch))

    def generate_name(self, value, epoch):
        return os.path.join(self.root, f"epoch-{epoch}-{value}.pth")


class GoogleDoodleTriplets(Dataset):
    """
    Load the MNIST dataset in pairs of similar(positive)/non-similar(negative) pairs
    """

    def __init__(self, dataset):
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

        # choose an image with the different label than the anchor
        # a = np.random.choice(np.setdiff1d(range(0, 10), label))
        # b = np.random.choice(self.class_idx[a])
        # negative, _ = self.dataset[b]
        negative, _ = self.dataset[
            np.random.choice(self.class_idx[np.random.choice(np.setdiff1d(range(self.dataset.num_classes()), label))])]
        return [anchor, positive, negative], label

    def __len__(self):
        return len(self.data)


def train_epoch(model, criterion, train_loader, optimizer):
    """
    Training of an epoch with Contrastive loss and training pairs
    model: network
    train_loader: train_loader loading pairs of positive/negative images and pair-label in batches.
    optimizer: optimizer to use in the training
    margin: loss margin
    """

    model.train()
    total_loss = 0
    loader = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()

        # extract descriptor for anchor and the corresponding pos/neg images
        output = [model(sample) for sample in data]

        # compute the contrastive loss

        loss = criterion(*output)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            loader.set_description(
                'Batch loss: {:.6f}'.format(loss.item()))
    print('Epoch average loss {:.6f}'.format(total_loss / len(train_loader.dataset)))
    return total_loss / len(train_loader.dataset)


if __name__ == '__main__':
    transforms = GoogleDoodleDataModule().default_transforms()

    dataset_test = GoogleDoodleDataset('google_doodle_dataset/data', train=False, transform=transforms, classes=9)

    # mnist dataset structure - train part
    dataset_train = GoogleDoodleDataset('google_doodle_dataset/data', train=True, transform=transforms, classes=9)

    # mnist dataset in positive/negative pairs structure
    dataset_train_triplets = GoogleDoodleTriplets(dataset_train)
    batch_size = 1024
    train_loader = DataLoader(dataset_train_triplets, batch_size=batch_size, shuffle=True)
    # loader of the test set (no pairs here)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = resnet18(True)  # initialize the network
    summary(model, (3, 32, 32))

    # %%

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.8)
    criterion = nn.TripletMarginLoss()
    saver = ModelSaver('resnet-triplets')
    metrics = []
    print('Training with Contrastive loss and training pairs')
    # train with contrastive loss
    for epoch in range(1, 10 + 1):
        print('Epoch {}'.format(epoch))
        loss = train_epoch(model, criterion, train_loader, optimizer)
        saver.update(model, loss, epoch)
        metrics.append(distance_validation(model, test_loader))
        scheduler.step()
