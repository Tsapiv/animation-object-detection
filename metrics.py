import numpy as np
import torch


def distance_validation(model, test_loader, step=100):
    """
    Compute accuracy on the test set
    model: network
    test_loader: test_loader loading images and labels in batches
    step: step to iterate over images (for faster evaluation)
    """

    model.eval()
    des = torch.Tensor()
    labels = torch.LongTensor()
    for batch_idx, (data, target) in enumerate(test_loader):
        des = torch.cat((des, model(data)))
        labels = torch.cat((labels, target))

    # compute all pair-wise distances
    cdistances = np.cdist(des.data.numpy(), des.data.numpy(), 'euclidean')

    # find rank of closest positive image (using each descriptor as a query)
    minrank_positive = []
    for i in range(0, len(cdistances), step):
        idx = np.argsort(cdistances[i])
        minrank_positive.append(np.min([j for (j, x) in enumerate(labels[idx[1:-1]]) if x == labels[i]]))
    return (np.array(minrank_positive) < 1).mean(), (np.array(minrank_positive) < 3).mean()