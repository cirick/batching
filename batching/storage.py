import os
from abc import ABC, abstractmethod
import datetime

import numpy as np
import boto3


class BatchStorage(ABC):
    def __init__(self, batch_meta):
        self._batch_meta = batch_meta

    @property
    def meta(self):
        return self._batch_meta

    @abstractmethod
    def save(self, X_batch, y_batch):
        raise NotImplementedError

    @abstractmethod
    def load(self, batch_id, validation=False):
        raise NotImplementedError


class BatchStorageFile(BatchStorage):
    def __init__(self, batch_meta, directory=None, validation_tag="v"):
        super(BatchStorageFile, self).__init__(batch_meta)
        self._path = directory
        self._validation_tag = validation_tag

        if not self._path:
            self._path = f"./cache/batches-{datetime.datetime.now():%Y-%m-%d-%H%M%S}"

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    @property
    def directory(self):
        return self._path

    def save(self, X_batch, y_batch, validation=False):
        filename = self.meta.save()
        file_location = f"{self._path}/{filename}.npz"
        np.savez(file_location, features=X_batch, labels=y_batch)
        return file_location

    def load(self, batch_id, validation=False):
        filename = self.meta.load(batch_id, validation)
        file_location = f"{self._path}/{filename}.npz"
        data = np.load(file_location)
        X = data["features"]
        y = data["labels"]
        return X, y


class BatchStorageS3(BatchStorage):
    def __init__(self, batch_meta, s3_bucket, s3_prefix="", validation_tag="v"):
        super(BatchStorageS3, self).__init__(batch_meta)
        s3 = boto3.resource("s3", region_name="us-east-1")
        self._bucket = s3.Bucket(s3_bucket)
        self._prefix = s3_prefix
        self._validation_tag = validation_tag

    def save(self, X_batch, y_batch):
        X_path, y_path = self.meta.save()
        np.savez(X_path, features=X_batch, labels=y_batch)
        self._bucket.upload_file(Filename=X_path, Key=seconds_key)

        self.meta.save(X_path, y_path)

    def load(self, batch_id, validation=False):
        X_path, y_path = self.meta.load(batch_id, validation)

        self._bucket.download_file(Key=seconds_key, Filename=X_path)
        self._bucket.download_file(Key=seconds_key, Filename=y_path)

        data = np.load(X_path)
        X = data["features"]
        y = data["labels"]

        return X, y
