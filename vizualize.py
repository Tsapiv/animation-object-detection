import cv2
import numpy as np
import torch
from matplotlib.pyplot import imsave, imshow
from pl_bolts.models.autoencoders import VAE
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from torchvision.utils import make_grid
from torchsummary import summary



if __name__ == '__main__':
    vae = VAE.load_from_checkpoint(
        checkpoint_path='google-exp-10/exp-epoch=49-val_loss=0.03.ckpt')
    vae.eval()
    summary(vae, (3, 32, 32))
    num_preds = 16

    shape = (num_preds, 1, 256)
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
    img = cv2.resize(img, (img.shape[1]*10, img.shape[0]*10), interpolation=cv2.INTER_NEAREST)
    imsave('pred.png', img)
    imshow(img)
