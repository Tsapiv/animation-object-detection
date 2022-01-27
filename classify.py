import argparse
import json

import numpy as np

from fabric import make_datamodule
from google_doodle_dataset import GoogleDoodleDataModule
from hardnet import HardNet
from triplets import ResNet18Encoder
from utils import mAP, clustering_accuracy
from utils import parse_config
from sklearn.manifold import TSNE
import plotly.express as px
from vae import WrappedVAE

MODELS = {
    'vea': WrappedVAE,
    'cnn': ResNet18Encoder,
    'hardnet': HardNet
}

if __name__ == '__main__':
    config = parse_config()
    model = MODELS[config['model_type']].load_from_checkpoint(checkpoint_path=config['checkpoint_path'],
                                                              map_location='cpu')
    model.eval()
    classes = np.array(list(json.load(open(f"{config['data_path']}/classes.json")).values()))
    batch_size = config['batch_size']
    datamodule: GoogleDoodleDataModule = make_datamodule(config['type'], config['data_path'], batch_size=batch_size)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    data, labels = next(iter(loader))
    X = model(data).detach()

    print("mAP:", mAP(X, labels))
    print("Clustering:", clustering_accuracy(model, loader, 'cpu', nth=1))
    X_emb = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(X.detach().numpy())
    fig = px.scatter_3d(x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2], color=classes[labels.detach().numpy()])
    fig.update_traces(marker_size=5)
    fig.show()
