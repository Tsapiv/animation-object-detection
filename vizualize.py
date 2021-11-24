import numpy as np
import torch
from matplotlib.pyplot import imsave, imshow
from pl_bolts.models.autoencoders import VAE
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from torchvision.utils import make_grid

if __name__ == '__main__':
    vae = VAE.load_from_checkpoint(
        checkpoint_path='mnist-exp/lightning_logs/version_7/checkpoints/epoch=29-step=22499.ckpt')
    shape = (32, 256)
    num_preds = 16
    p = torch.distributions.Normal(torch.zeros(*shape), torch.ones(*shape))
    z = p.rsample()

    # SAMPLE IMAGES
    with torch.no_grad():
        pred = vae.decoder(z.to(vae.device)).cpu()

    # UNDO DATA NORMALIZATION
    normalize = emnist_normalization('mnist')
    mean, std = np.array(normalize.mean), np.array(normalize.std)
    img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

    # PLOT IMAGES
    imsave('pred.png', img)
    imshow(img)
