from google_doodle_dataset import GoogleDoodleDataModule
from mnist_dataset import EMNISTDataModule
from pl_bolts.datamodules import CIFAR10DataModule

MODULES = {
    'mnist': EMNISTDataModule,
    'cifar': CIFAR10DataModule,
    'google': GoogleDoodleDataModule
}


def make_datamodule(module_type: str, *args, **kwargs):
    return MODULES[module_type](*args, **kwargs)
