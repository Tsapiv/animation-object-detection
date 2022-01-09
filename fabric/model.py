from triplets import ResNet18Encoder
from vae.model import WrappedVAE

MODELS = {
    'google-doodle': WrappedVAE,
    'google-doodle-wrapped': ResNet18Encoder
}


def make_model(model_type: str, *args, **kwargs):
    return MODELS[model_type](*args, **kwargs)
