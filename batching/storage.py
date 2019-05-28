import os
from abc import ABC, abstractmethod
import datetime

import json
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

    @abstractmethod
    def save_meta(self, params):
        raise NotImplementedError

    @abstractmethod
    def load_meta(self):
        raise NotImplementedError


class BatchStorageMemory(BatchStorage):
    def __init__(self, batch_meta, validation_tag="v"):
        super().__init__(batch_meta)
        self._validation_tag = validation_tag
        self._data = {}
        self.meta_data = None

    def save(self, X_batch, y_batch):
        filename = self.meta.save()
        self._data[filename] = {"features": X_batch, "labels": y_batch}
        return filename

    def save_meta(self, params):
        meta_params = self.meta.get_meta_params()
        params.update(meta_params)
        self.meta_data = params

    def load_meta(self):
        if self.meta_data is None:
            raise Exception("No saved metadata")

        params = self.meta_data
        params["train_map"] = {int(k): v for k, v in params["train_map"].items()}
        params["val_map"] = {int(k): v for k, v in params["val_map"].items()}
        self.meta.set_meta_params(train_ids=params["train_ids"],
                                  train_map=params["train_map"],
                                  validation_ids=params["val_ids"],
                                  validation_map=params["val_map"])
        return params

    def load(self, batch_id, validation=False):
        filename = self.meta.load(batch_id, validation)
        X = self._data[filename]["features"]
        y = self._data[filename]["labels"]
        return X, y


class BatchStorageFile(BatchStorage):
    def __init__(self, batch_meta, directory=None, validation_tag="v"):
        super().__init__(batch_meta)
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

    def save_meta(self, params):
        meta_params = self.meta.get_meta_params()
        params.update(meta_params)
        with open(f"{self._path}/meta.json", 'w') as outfile:
            outfile.write(json.dumps(params))

    def load_meta(self):
        with open(f"{self._path}/meta.json", 'r') as infile:
            params = json.load(infile)

        params["train_map"] = {int(k): v for k, v in params["train_map"].items()}
        params["val_map"] = {int(k): v for k, v in params["val_map"].items()}
        self.meta.set_meta_params(train_ids=params["train_ids"],
                                  train_map=params["train_map"],
                                  validation_ids=params["val_ids"],
                                  validation_map=params["val_map"])
        return params

    def load(self, batch_id, validation=False):
        filename = self.meta.load(batch_id, validation)
        file_location = f"{self._path}/{filename}.npz"
        data = np.load(file_location)
        X = data["features"]
        y = data["labels"]
        return X, y


class BatchStorageS3(BatchStorage):
    def __init__(self, batch_meta, s3_bucket, s3_prefix="", validation_tag="v"):
        super().__init__(batch_meta)
        s3 = boto3.resource("s3", region_name="us-east-1")
        self._bucket = s3.Bucket(s3_bucket)
        self._prefix = s3_prefix
        self._validation_tag = validation_tag

    def save(self, X_batch, y_batch):
        filename = self.meta.save()
        file_location = f"{filename}.npz"
        np.savez(file_location, features=X_batch, labels=y_batch)

        self._bucket.upload_file(Filename=file_location, Key=f"{self._prefix}/{filename}")

    def save_meta(self, params):
        meta_params = self.meta.get_meta_params()
        params.update(meta_params)
        with open("meta.json", 'w') as outfile:
            outfile.write(json.dumps(params))

        self._bucket.upload_file(Filename="meta.json", Key=f"{self._prefix}/meta.json")

    def load_meta(self):
        self._bucket.download_file(Key=f"{self._prefix}/meta.json", Filename="meta.json")

        with open(f"meta.json", 'r') as infile:
            params = json.load(infile)

        params["train_map"] = {int(k): v for k, v in params["train_map"].items()}
        params["val_map"] = {int(k): v for k, v in params["val_map"].items()}
        self.meta.set_meta_params(train_ids=params["train_ids"],
                                  train_map=params["train_map"],
                                  validation_ids=params["val_ids"],
                                  validation_map=params["val_map"])
        return params

    def load(self, batch_id, validation=False):
        filename = self.meta.load(batch_id, validation)

        self._bucket.download_file(Key=f"{self._prefix}/{filename}", Filename=filename)

        data = np.load(filename)
        X = data["features"]
        y = data["labels"]

        return X, y
