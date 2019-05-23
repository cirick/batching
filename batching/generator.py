import tensorflow as tf
import numpy as np


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_meta_ids, storage, seed=None, n_classes=2):
        self._batch_ids = batch_meta_ids.ids
        self._batch_map = batch_meta_ids.map
        self._storage = storage
        self._n_classes = n_classes

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
        batch_idx = self.indexes[index]

        X_file = self._batch_map[batch_idx]["X"]
        y_file = self._batch_map[batch_idx]["y"]

        X, y = self._storage.load(X_file, y_file)

        if self._n_classes > 2:
            y = tf.keras.utils.to_categorical(y, self._n_classes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self._batch_ids
        np.random.shuffle(self.indexes)
