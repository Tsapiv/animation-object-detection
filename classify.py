import numpy as np
import torch
from matplotlib.pyplot import imsave, imshow
from pl_bolts.models.autoencoders import VAE
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from torchvision.utils import make_grid
from torchsummary import summary
import matplotlib.pyplot as plt

from fabric import make_datamodule
from google_doodle_dataset import GoogleDoodleDataModule
from utils import parse_config
from tsne_torch import TorchTSNE as TSNE

if __name__ == '__main__':
    vae = VAE.load_from_checkpoint(
        checkpoint_path='google-exp/exp-epoch=48-val_loss=0.03.ckpt')

    config = parse_config()
    batch_size = 512
    datamodule: GoogleDoodleDataModule = make_datamodule(config['type'], config['data_path'], batch_size=batch_size)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    data = next(iter(loader))
    print(data[0].shape, data[1].shape)
    vae.eval()
    # summary(vae, (3, 32, 32))
    base = vae.encoder(data[0])
    X = torch.concat([vae.fc_mu(base), vae.fc_var(base)], dim=-1)
    shape = (1, 3, 32, 32)
    X_emb = TSNE(n_components=2, perplexity=30, n_iter=3000, verbose=True, initial_dims=batch_size).fit_transform(X)  # returns shape (n_samples, 2)
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=data[1])
    plt.savefig("3-classes-tsne-train.png", dpi=300)
    plt.show()
    # print(X_emb.shape, X_emb)
    # num_preds = 16
    # p = torch.distributions.Normal(torch.zeros(*shape), torch.ones(*shape))
    # z = p.rsample()
    # a = vae.encoder(z)
    # b = vae.fc_mu(a)
    # print(a.shape, b.shape)

