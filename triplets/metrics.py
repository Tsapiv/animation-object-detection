import numpy as np
import torch


def distance_validation(model, test_loader, step=100, device='cuda:0'):
    """
    Compute accuracy on the test set
    model: network
    test_loader: test_loader loading images and labels in batches
    step: step to iterate over images (for faster evaluation)
    """

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
