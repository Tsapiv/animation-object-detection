import argparse

import cv2
import numpy as np
import torch
from matplotlib.pyplot import imsave, imshow
from pl_bolts.models.autoencoders import VAE
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from torchvision.utils import make_grid
from torchsummary import summary


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--prediction-number', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_options()
    vae = VAE.load_from_checkpoint(
        checkpoint_path=parameters.checkpoint_path)
    vae.eval()
    parameters.verbose and summary(vae, (3, 32, 32))

    shape = (parameters.prediction_number, 1, 256)
    p = torch.distributions.Normal(torch.zeros(*shape), torch.ones(*shape))
    z = p.rsample()

    # SAMPLE IMAGES
    with torch.no_grad():
        prediction = vae.decoder(z.to(vae.device)).cpu()

    # UNDO DATA NORMALIZATION
    normalize = emnist_normalization('mnist')
    mean, std = np.array(normalize.mean), np.array(normalize.std)
    img = make_grid(prediction).permute(1, 2, 0).numpy() * std + mean

    # PLOT IMAGES
    img = cv2.resize(img[:, :, 0], (img.shape[1] * 10, img.shape[0] * 10))
    imsave(f'{parameters.output_dir}/prediction-{parameters.prediction_number}.png', img)
    parameters.verbose and imshow(img)
