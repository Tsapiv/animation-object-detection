import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from pl_bolts.models import VAE

from fabric import make_datamodule
from google_doodle_dataset import GoogleDoodleDataModule
from triplets import ResNet18Encoder, mAP, clustering_accuracy
from utils import parse_config
from sklearn.manifold import TSNE
import plotly.express as px

from vae.model import WrappedVAE

if __name__ == '__main__':
    # vae = VAE.load_from_checkpoint(
    #     checkpoint_path='exp/google-exp-9/exp-epoch=49-val_loss=0.03.ckpt')
    # vae = ResNet18Encoder.load_from_checkpoint(checkpoint_path='exp/google-exp-10-wrapped/exp-epoch=49-val_loss=0.00.ckpt', map_location='cpu')
    vae = WrappedVAE()#WrappedVAE.load_from_checkpoint(checkpoint_path='exp/google-exp-10/exp-epoch=46-val_loss=0.03.ckpt', map_location='cpu')

    vae.eval()
    config = parse_config()
    classes = np.array(list(json.load(open(f"{config['data_path']}/classes.json")).values()))
    batch_size = 1024
    datamodule: GoogleDoodleDataModule = make_datamodule(config['type'], config['data_path'], batch_size=batch_size)
    datamodule.setup()
    loader = datamodule.test_dataloader()
    data, labels = next(iter(loader))
    X = vae.encoder(data).detach()
    print("mAP:", mAP(X, labels))
    print("Clustering:", clustering_accuracy(vae.encoder, loader, 'cpu', nth=20))
    # X_emb = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(X.detach().numpy())
    #
    # fig = px.scatter_3d(x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2], color=classes[labels.detach().numpy()])
    # fig.update_traces(marker_size=5)
    # fig.show()
