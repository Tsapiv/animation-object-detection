import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from google_doodle_dataset import GoogleDoodleDataset
from .dataset_wrapper import TripletsDatasetWrapper
from .metrics import distance_validation
from .model import ResNet18Encoder
from utils import ModelSaver, default_transform, parse_config


class Trainer:
    def __init__(self, config: Dict, model: nn.Module):
        self.__device = torch.device(config.get('device', 'cpu'))
        self.__transforms = default_transform()

        dataset_test = GoogleDoodleDataset(config['root'], train=False, transform=self.__transforms,
                                           classes=config['classes'])
        dataset_train = GoogleDoodleDataset(config['root'], train=True, transform=self.__transforms,
                                            classes=config['classes'])
        dataset_train_triplets = TripletsDatasetWrapper(dataset_train)

        self.__train_loader = DataLoader(dataset_train_triplets, batch_size=config.get('batch_size'), shuffle=True,
                                         pin_memory=True)
        self.__test_loader = DataLoader(dataset_test, batch_size=config.get('batch_size'), shuffle=False,
                                        pin_memory=True)
        self.__model = model
        self.__optimizer = optim.SGD(self.__model.parameters(), lr=config.get('lr', 0.001),
                                     momentum=config.get('momentum', 0.9))
        self.__scheduler = lr_scheduler.StepLR(self.__optimizer, 1, gamma=config.get('gamma', 0.9))
        self.__criterion = nn.TripletMarginLoss()
        self.__save_dir = config.get('save_dir', 'triplets')
        self.__saver = ModelSaver(config.get('save_dir', os.path.join(self.__save_dir, 'models')))
        self.__epochs = config['epochs']

    def train_epoch(self):
        self.__model.train()
        total_loss = 0
        loader = tqdm(self.__train_loader)
        for batch_idx, (data, target) in enumerate(loader):
            self.__optimizer.zero_grad()

            # extract descriptor for anchor and the corresponding pos/neg images
            output = [self.__model(sample.to(self.__device)) for sample in data]

            # compute the contrastive loss

            loss = self.__criterion(*output)

            loss.backward()
            self.__optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                loader.set_description(
                    'Epoch average loss {:.6f}'.format(total_loss / len(self.__train_loader)))
        return total_loss / len(self.__train_loader)

    def train(self):
        metrics = []
        for epoch in range(1, self.__epochs + 1):
            print('Epoch {}'.format(epoch))
            loss = self.train_epoch()
            self.__saver.update(self.__model, loss, epoch)
            metrics.append(distance_validation(self.__model, self.__test_loader))
            self.__scheduler.step()
            np.save(os.path.join(self.__save_dir, 'training.npy'), np.array(metrics))


if __name__ == '__main__':
    config = parse_config()
    model = ResNet18Encoder(input_height=32)
    trainer = Trainer(config, model)
    trainer.train()
