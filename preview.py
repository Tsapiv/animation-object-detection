import os
from os import listdir

import numpy as np
import cv2
from google_doodle_dataset import GoogleDoodleDataset

if __name__ == '__main__':
    root = 'google_doodle_dataset/raw/'

    data = [np.load(os.path.join(root, f)) for f in listdir(root) if f.endswith('.npy')]
    targets = [np.full(len(category), idx) for idx, category in enumerate(data)]
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i, lab in enumerate(targets):
        idx = np.arange(0, len(lab))
        np.random.shuffle(idx)
        delim = int(len(idx) * 0.9)
        train_idx, test_idx = idx[:delim], idx[delim:]
        train_data.append(data[i][train_idx])
        train_label.append(lab[train_idx])
        test_data.append(data[i][test_idx])
        test_label.append(lab[test_idx])

    np.save('google_doodle_dataset/data/10-classes-train-data.npy', np.reshape(np.vstack(train_data), (-1, 28, 28)))
    np.save('google_doodle_dataset/data/10-classes-train-labels.npy', np.hstack(train_label))
    np.save('google_doodle_dataset/data/10-classes-test-data.npy', np.reshape(np.vstack(test_data), (-1, 28, 28)))
    np.save('google_doodle_dataset/data/10-classes-test-labels.npy', np.hstack(test_label))

