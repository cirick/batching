import pandas as pd
import numpy as np
from nose import tools

from batching.builder import Builder
from batching.storage import BatchStorageMemory
from batching.storage_meta import StorageMeta


def test_builder_config():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(32)), unit="s"),
                                     "A": np.ones(32),
                                     "B": np.ones(32),
                                     "y": np.ones(32)})
                       for _ in range(1)]

    look_back = 2
    look_forward = 1
    batch_seconds = 1

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=16)

    batch_generator.generate_and_save_batches(feature_df_list)

    tools.eq_(batch_generator.batch_size, 16)
    tools.eq_(batch_generator._features, list(feature_set))
    tools.eq_(batch_generator._look_forward, 1)
    tools.eq_(batch_generator._look_back, 2)
    tools.eq_(batch_generator._n_seconds, 1)
    assert not batch_generator._stratify


def test_builder_storage_meta():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(35)), unit="s"),
                                     "A": np.ones(35),
                                     "B": np.ones(35),
                                     "y": np.ones(35)})
                       for _ in range(1)]

    look_back = 2
    look_forward = 1
    batch_seconds = 1

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=16)

    batch_generator.generate_and_save_batches(feature_df_list)

    tools.eq_(len(meta.train.ids), 2)
    tools.eq_(len(meta.validation.ids), 0)


def test_builder_storage_meta_validation():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(35)), unit="s"),
                                     "A": np.ones(35),
                                     "B": np.ones(35),
                                     "y": np.ones(35)})
                       for _ in range(1)]

    look_back = 2
    look_forward = 1
    batch_seconds = 1

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=16)

    batch_generator.generate_and_save_batches(feature_df_list)

    tools.eq_(len(meta.train.ids), 1)
    tools.eq_(len(meta.validation.ids), 1)


def test_builder_stratify():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(160)), unit="s"),
                                     "A": np.ones(160),
                                     "B": np.ones(160),
                                     "y": np.ones(160)})
                       for _ in range(1)]

    look_back = 0
    look_forward = 0
    batch_seconds = 1

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=16,
                              stratify_nbatch_groupings=3,
                              pseudo_stratify=True)

    batch_generator.generate_and_save_batches(feature_df_list)

    assert batch_generator._stratify
    tools.eq_(len(meta.train.ids), 5)
    tools.eq_(len(meta.validation.ids), 5)
