from triplets import ResNet18Encoder
from hardnet import HardNet
from vae.model import WrappedVAE

MODELS = {
    'google-doodle': WrappedVAE,
    'google-doodle-wrapped': ResNet18Encoder,
    'hardnet': HardNet
}


def make_model(model_type: str, *args, **kwargs):
    return MODELS[model_type](*args, **kwargs)
