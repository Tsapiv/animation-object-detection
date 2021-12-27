import os
from os import listdir

import numpy as np
import cv2
import argparse
import json


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--skip-mapping', action='store_false')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_options()
    root = parameters.raw_data
    out = os.path.join(parameters.output_dir, 'data')
    os.makedirs(out, exist_ok=True)
    filenames = [os.path.join(root, f) for f in listdir(root) if f.endswith('.npy')]
    if parameters.skip_mapping:
        json.dump(dict(enumerate(map(lambda x: x.split('.')[0].split('_')[-1], filenames))),
                  open(os.path.join(out, 'classes.json'), 'w'),
                  indent=4)
    data = list(map(np.load, filenames))
    targets = [np.full(len(category), idx) for idx, category in enumerate(data)]
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i, lab in enumerate(targets):
        idx = np.arange(0, len(lab))
        np.random.shuffle(idx)
        delim = int(len(idx) * parameters.split_ratio)
        train_idx, test_idx = idx[:delim], idx[delim:]
        train_data.append(data[i][train_idx])
        train_label.append(lab[train_idx])
        test_data.append(data[i][test_idx])
        test_label.append(lab[test_idx])

    np.save(f'{out}/{parameters.name}-train-data.npy', np.reshape(np.vstack(train_data), (-1, 28, 28)))
    np.save(f'{out}/{parameters.name}-train-labels.npy', np.hstack(train_label))
    np.save(f'{out}/{parameters.name}-test-data.npy', np.reshape(np.vstack(test_data), (-1, 28, 28)))
    np.save(f'{out}/{parameters.name}-test-labels.npy', np.hstack(test_label))
