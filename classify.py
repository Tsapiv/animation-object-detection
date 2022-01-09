# TODO: make separate files for each experiment

import numpy as np
import plotly.express as px
import torch
from matplotlib.pyplot import imsave, imshow
from pl_bolts.models.autoencoders import VAE, resnet18_encoder
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
from tqdm import tqdm

from fabric import make_datamodule
from google_doodle_dataset import GoogleDoodleDataModule, GoogleDoodleDataset
from triplets import mAP, clustering_accuracy
from utils import parse_config, default_transform
# from tsne_torch import TorchTSNE as TSNE
from plotly import graph_objects as go

if __name__ == '__main__':
    # class_map = np.array(['airplane', 'bicycle', 'bird', 'blueberry', 'book', 'bulldozer', 'cat', 'crab', 'hand', 'octagon'])
    class_map = np.array(['known'] * 9 + ['unknown'])

    vae = torch.load('exp/real-resnet18-triplets-exp-9/epoch-10-2.7731600012364955e-08.pth', map_location='cpu')#VAE.load_from_checkpoint(checkpoint_path='exp/google-exp-9/exp-epoch=49-val_loss=0.03.ckpt', map_location='cpu')#
    vae.eval()

    config = parse_config()
    batch_size = 64
    extra_size = 100
    transforms = default_transform()
    # extra_class = torch.stack(
    #     [transforms(bit) for bit in np.reshape(np.load('full_numpy_bitmap_octagon.npy')[:extra_size], (-1, 28, 28))])
    # extra_labels = torch.full((extra_size,), 9)
    label_mask = ['circle'] * batch_size + ['square'] * extra_size
    dataset_test = GoogleDoodleDataset('google_doodle_dataset/google-doodle-9/data', train=False, transform=transforms, classes=9, ratio=0.02)
    loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)
    # knn = KMeans(n_clusters=9)
    # correct = 0
    # total = 0
    # l = tqdm(loader)
    # for data, labels in l:
    #     # base = vae.encoder(data)
    #     # X = torch.concat([vae.fc_mu(base), vae.fc_var(base)], dim=-1)
    #     X = vae(data)
    #     X_train, X_test, y_train, y_test = train_test_split(X.detach().numpy(), labels.numpy(),
    #                                                         test_size=0.3)  # 70% training and 30% test
    #     # for i in range(2, 25):
    #
    #
    #     # Train the model using the training sets
    #     knn.fit(X_train, y_train)
    #
    #     # Predict the response for test dataset
    #     y_pred = knn.predict(X_test)
    #     correct += np.sum(y_pred == y_test)
    #     total += len(y_pred)
    #     l.set_description(f"Accuracy: {correct/ total}")
    #
    # print(correct/total)

    # summary(vae, (3, 32, 32))
    # print(len(loader))
    print(clustering_accuracy(vae, loader, 'cpu', nth=1))

    data, labels = next(iter(loader))
    # data = torch.cat((data, extra_class), dim=0)
    # labels = torch.cat((labels, extra_labels), dim=-1)
    print(data.shape, labels.shape)

    # X = vae(data)
    X = vae.encoder(data)
    # X = torch.concat([vae.fc_mu(base), vae.fc_var(base)], dim=-1)
    # X_train, X_test, y_train, y_test = train_test_split(X.detach().numpy(), labels.numpy(),
    #                                                   test_size=0.3)  # 70% training and 30% test
    # # # for i in range(2, 25):
    # knn = KMeans(n_clusters=9)
    # #
    # # # Train the model using the training sets
    # knn.fit(X_train)
    # #
    # # # Predict the response for test dataset
    # correct = 0
    # total = 0
    # for class_ in range(9):
    #     indexes = np.argwhere(y_test == class_)
    #     selected = X_test[indexes]
    #     if len(selected) == 0:
    #         continue
    #     y_pred = knn.predict(np.squeeze(selected, axis=1))
    #     _, counts = np.unique(y_pred, return_counts=True)
    #     correct += max(counts)
    #     total += len(y_pred)
    # print(correct/total)


    # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # shape = (1, 3, 32, 32)
    X_emb = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(X.detach().numpy())

    # fig = px.scatter(x=X_emb[:, 0], y=X_emb[:, 1], color=class_map[labels.detach().numpy()])
    # fig.update_traces(marker_size=8)
    # fig.write_image('figure2.png', scale=4)
    # fig.show()

    fig = px.scatter_3d(x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2], color=class_map[labels.detach().numpy()], symbol=label_mask)
    fig.update_traces(marker_size=5)
    # fig.write_image('figure.png', scale=4)
    fig.show()

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter3d(x=X_emb[:, 0], y=X_emb[:, 1], z=X_emb[:, 2],
    #                  mode='markers',
    #                  marker=dict(
    #                      size=5,
    #                      color=labels,  # set color to an array/list of desired values
    #                      symbol=label_mask,
    #                      # colorscale='Viridis',  # choose a colorscale
    #                      opacity=0.8)
    #                  )
    # )
    #
    #
    # fig.show()

    # plt.scatter(X_emb[:, 0], X_emb[:, 1], c=data[1])
    # plt.savefig("10-classes-tsne-train.png", dpi=300)
    # plt.show()

    # print(X_emb.shape, X_emb)
    # num_preds = 16
    # p = torch.distributions.Normal(torch.zeros(*shape), torch.ones(*shape))
    # z = p.rsample()
    # a = vae.encoder(z)
    # b = vae.fc_mu(a)
    # print(a.shape, b.shape)
