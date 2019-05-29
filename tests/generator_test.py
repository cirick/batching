import pandas as pd
import numpy as np
from nose import tools

from batching.builder import Builder
from batching.generator import BatchGenerator
from batching.storage import BatchStorageMemory
from batching.storage_meta import StorageMeta


def test_generator():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(32)), unit="s"),
                                     "A": np.ones(32),
                                     "B": np.ones(32),
                                     "y": np.ones(32)})
                       for _ in range(1)]

    look_back = 2
    look_forward = 2
    batch_seconds = 1

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=16)

    batch_generator.generate_and_save_batches(feature_df_list)
    train_generator = BatchGenerator(storage, is_validation=False, seed=42)

    X, y = train_generator[0]
    tools.eq_(X.shape, (16, 5, 2))
    assert np.array_equal(X, np.zeros(X.shape))


def test_validation_gen():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(64)), unit="s"),
                                     "A": np.ones(64),
                                     "B": np.ones(64),
                                     "y": np.ones(64)})
                       for _ in range(1)]

    look_back = 0
    look_forward = 0
    batch_seconds = 1

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=8)

    batch_generator.generate_and_save_batches(feature_df_list)
    validation_generator = BatchGenerator(storage, is_validation=True)

    X, y = validation_generator[0]
    tools.eq_(X.shape, (8, 1, 2))
    tools.eq_(len([(x, y) for x, y in validation_generator]), 4)
    assert np.array_equal(X, np.zeros(X.shape))


def test_validation_gen_window():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(70)), unit="s"),
                                     "A": np.ones(70),
                                     "B": np.ones(70),
                                     "y": np.ones(70)})
                       for _ in range(1)]

    look_back = 6
    look_forward = 0
    batch_seconds = 1

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=8)

    batch_generator.generate_and_save_batches(feature_df_list)
    validation_generator = BatchGenerator(storage, is_validation=True)

    X, y = validation_generator[0]
    tools.eq_(X.shape, (8, 7, 2))
    tools.eq_(y.shape, (8, ))
    tools.eq_(len([(x, y) for x, y in validation_generator]), 4)
    assert np.array_equal(X, np.zeros(X.shape))


def test_validation_gen_window_categorical():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(70)), unit="s"),
                                     "A": np.ones(70),
                                     "B": np.ones(70),
                                     "y": np.ones(70)})
                       for _ in range(1)]

    look_back = 6
    look_forward = 0
    batch_seconds = 1

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=8)

    batch_generator.generate_and_save_batches(feature_df_list)

    n_categories = 4
    validation_generator = BatchGenerator(storage, is_validation=True, n_classes=n_categories)

    X, y = validation_generator[0]
    tools.eq_(X.shape, (8, 7, 2))
    tools.eq_(y.shape, (8, n_categories))
    tools.eq_(len([(x, y) for x, y in validation_generator]), 4)
    assert np.array_equal(X, np.zeros(X.shape))
