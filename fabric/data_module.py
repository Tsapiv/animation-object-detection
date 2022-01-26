from google_doodle_dataset import GoogleDoodleDataModule
from mnist_dataset import EMNISTDataModule
from pl_bolts.datamodules import CIFAR10DataModule
from triplets.data_module import GoogleDoodleTripletsDataModule

MODULES = {
    'mnist': EMNISTDataModule,
    'cifar': CIFAR10DataModule,
    'google-doodle': GoogleDoodleDataModule,
    'google-doodle-wrapped': GoogleDoodleTripletsDataModule,
    'hardnet': GoogleDoodleTripletsDataModule
}


def make_datamodule(module_type: str, *args, **kwargs):
    return MODULES[module_type](*args, **kwargs)
