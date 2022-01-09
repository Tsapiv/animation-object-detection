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
from dataset_wrapper import TripletsDatasetWrapper
from metrics import rank_validation, mAP
from model import ResNet18Encoder
from utils import ModelSaver, default_transform, parse_config
from torchsummary import summary

class Trainer:
    def __init__(self, config: Dict, model: nn.Module):
        self.__device = torch.device(config.get('device', 'cpu'))
        self.__transforms = default_transform()

        dataset_test = GoogleDoodleDataset(config['root'], train=False, transform=self.__transforms,
                                           classes=config['classes'], ratio=0.001)
        dataset_train = GoogleDoodleDataset(config['root'], train=True, transform=self.__transforms,
                                            classes=config['classes'], ratio=0.0005)
        dataset_train_triplets = TripletsDatasetWrapper(dataset_train)

        self.__train_loader = DataLoader(dataset_train_triplets, batch_size=config.get('batch_size'), shuffle=True,
                                         pin_memory=True)
        self.__test_loader = DataLoader(dataset_test, batch_size=config.get('batch_size'), shuffle=True,
                                        pin_memory=True)
        self.__model = model
        self.__optimizer = optim.SGD(self.__model.parameters(), lr=config.get('lr', 0.001),
                                     momentum=config.get('momentum', 0.9))
        self.__scheduler = lr_scheduler.StepLR(self.__optimizer, 1, gamma=config.get('gamma', 0.9))
        self.__criterion = nn.TripletMarginLoss()
        self.__save_dir = config.get('save_dir', 'triplets')
        self.__saver = ModelSaver(os.path.join(self.__save_dir, 'models'))
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
            metrics.append(mAP(self.__model, self.__test_loader, self.__device, nth=1).item())
            print(metrics[-1])
            self.__scheduler.step()
            np.save(os.path.join(self.__save_dir, 'training.npy'), np.array(metrics))


if __name__ == '__main__':
    vae = ResNet18Encoder().from_pretrained('cifar10-resnet18')
    summary(vae, (3, 32, 32))
    # config = parse_config()
    # model = ResNet18Encoder(input_height=32)
    # trainer = Trainer(config, model)
    # trainer.train()
