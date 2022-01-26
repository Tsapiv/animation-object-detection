import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import RetrievalMAP


def rank_validation(model, test_loader, step=100, device='cuda:0'):
    model.eval()
    des = torch.FloatTensor().to(device)
    labels = torch.LongTensor()
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % step == 0:
            des = torch.cat((des, model(data.to(device))))
            labels = torch.cat((labels, target))

    # compute all pair-wise distances
    cdistances = torch.cdist(des, des).cpu().detach().numpy()

    # find rank of closest positive image (using each descriptor as a query)
    minrank_positive = []
    for i in range(0, len(cdistances), 1):
        idx = np.argsort(cdistances[i])
        minrank_positive.append(np.min([j for (j, x) in enumerate(labels[idx[1:-1]]) if x == labels[i]]))
    minrank1, minrank3 = (np.array(minrank_positive) < 1).mean(), (np.array(minrank_positive) < 3).mean()
    return minrank1, minrank3


def mAP(descriptor, labels):
    # compute all pair-wise distances

    cdistances = -torch.cdist(descriptor, descriptor)
    metric = RetrievalMAP()
    # find rank of closest positive image (using each descriptor as a query)
    for idx, prediction in enumerate(cdistances):
        target: torch.Tensor = (labels == labels[idx])
        indeces = torch.LongTensor(len(labels)).fill_(idx)
        metric.update(prediction, target, indeces)
    return metric.compute()


def clustering_accuracy(model: Module, test_loader: DataLoader, device, nth=100, test_size=0.3, classes=10):
    model.eval()
    X = torch.FloatTensor().to(device)
    labels = torch.LongTensor()
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == len(test_loader) // nth:
            break
        X = torch.cat((X, model(data.to(device)).detach()))
        labels = torch.cat((labels, target))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X.detach().numpy(), labels.numpy(),
                                                        test_size=test_size)
    km = KMeans(n_clusters=classes)
    km.fit(X_train)
    correct = 0
    total = 0
    for class_ in range(classes):
        indexes = np.argwhere(y_test == class_)
        selected = X_test[indexes]
        if len(selected) == 0:
            continue
        y_pred = km.predict(np.squeeze(selected, axis=1))
        _, counts = np.unique(y_pred, return_counts=True)
        correct += max(counts)
        total += len(y_pred)
    return correct / total
