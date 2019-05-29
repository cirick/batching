import tensorflow as tf
import numpy as np


def to_categorical(y, n_classes):
    return np.eye(n_classes)[y.astype(int)]


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, storage, is_validation, seed=None, n_classes=1):
        self._storage = storage
        self._n_classes = n_classes
        self._batch_ids = storage.meta.get_ids(is_validation)
        self._is_validation = is_validation

        if seed:
            np.random.seed(seed)

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self._batch_ids)

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        batch_id = self.indexes[index]
        X, y = self._storage.load(batch_id, self._is_validation)

        if self._n_classes > 1:
            y = to_categorical(y, self._n_classes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self._batch_ids
        np.random.shuffle(self.indexes)
