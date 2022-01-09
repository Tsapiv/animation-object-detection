from abc import ABC

from google_doodle_dataset import GoogleDoodleDataset, GoogleDoodleDataModule
from .dataset_wrapper import TripletsDatasetWrapper


class GoogleDoodleTripletsDataModule(GoogleDoodleDataModule, ABC):
    name = "google-doodle-wrapped"
    dataset_cls = lambda *args, **kwargs: TripletsDatasetWrapper(GoogleDoodleDataset(*args[1:], **kwargs))
