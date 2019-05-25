import os
from abc import ABC, abstractmethod
import datetime
import numpy as np


class BatchStorage(ABC):
    @abstractmethod
    def save(self, sequence_number, X_batch, y_batch, validation=False):
        raise NotImplementedError

    @abstractmethod
    def load(self, X_file, y_file):
        raise NotImplementedError


class BatchStorageFile(BatchStorage):
    def __init__(self, directory=None, validation_tag="v"):
        self._path = directory
        self._validation_tag = validation_tag

        if not self._path:
            self._path = f"./cache/batches-{datetime.datetime.now():%Y-%m-%d-%H%M%S}"

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    @property
    def directory(self):
        return self._path

    def save(self, sequence_number, X_batch, y_batch, validation=False):
        val_tag = "" if not validation else self._validation_tag
        X_path = f"{self._path}/X{val_tag}_{sequence_number}.npy"
        y_path = f"{self._path}/y{val_tag}_{sequence_number}.npy"

        np.save(X_path, X_batch)
        np.save(y_path, y_batch)
        return X_path, y_path

    def load(self, X_path, y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y