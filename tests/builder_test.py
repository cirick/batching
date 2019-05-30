import pandas as pd
import numpy as np
from nose import tools
import datetime
from functools import reduce
from operator import add

from batching.builder import Builder
from batching.storage import BatchStorageMemory
from batching.storage_meta import StorageMeta


@tools.raises(Exception)
def test_no_dataset():
    batch_generator = Builder(BatchStorageMemory(StorageMeta()), [], 0, 0, 1)
    batch_generator.generate_and_save_batches([])


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
    batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds,
                              batch_size=16, verbose=True)

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


def test_remove_anchors():
    y = np.zeros(10)
    y[[1, 5, 7]] = 1
    now = int(datetime.datetime.timestamp(datetime.datetime.now()))
    df = pd.DataFrame({"time": pd.to_datetime(list(range(now, now + 10)), unit="s"),
                       "A": np.ones(10),
                       "B": np.ones(10),
                       "y": y})
    df.index = df["time"]

    batch_generator = Builder(BatchStorageMemory(StorageMeta()),
                              features=["A", "B"],
                              look_back=0, look_forward=0, n_seconds=1, batch_size=16,
                              stratify_nbatch_groupings=3,
                              pseudo_stratify=True)

    assert df["y"].any()
    batch_generator._remove_false_anchors(df, "y")
    assert not df["y"].any()


def test_normalize_on():
    feature_df_list = reduce(add, [[pd.DataFrame({"time": pd.to_datetime(list(range(50)), unit="s"),
                                                  "A": range(1, 51),
                                                  "B": range(101, 151),
                                                  "y": np.ones(50)}),
                                    pd.DataFrame({"time": pd.to_datetime(list(range(50)), unit="s"),
                                                  "A": range(51, 101),
                                                  "B": range(151, 201),
                                                  "y": np.ones(50)})]
                                   for _ in range(5)], [])

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage,
                              features=["A", "B"],
                              look_back=0, look_forward=0, n_seconds=1, batch_size=10,
                              normalize=True,
                              pseudo_stratify=False)

    batch_generator.generate_and_save_batches(feature_df_list)

    tools.assert_almost_equal(batch_generator.scaler.mean_[0], 50, delta=1)
    tools.assert_almost_equal(batch_generator.scaler.mean_[1], 150, delta=1)

    for batch in storage._data.values():
        # all batches have monotonically increasing numbers (range used to create data)
        assert np.diff(batch["features"][:, 0, 0]).all()  # feature A
        assert np.diff(batch["features"][:, 0, 1]).all()  # feature B


def test_normalize_off():
    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(160)), unit="s"),
                                     "A": range(160),
                                     "B": range(160),
                                     "y": np.ones(160)})
                       for _ in range(1)]

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage,
                              features=["A", "B"],
                              look_back=0, look_forward=0, n_seconds=1, batch_size=16,
                              normalize=False,
                              pseudo_stratify=False)

    batch_generator.generate_and_save_batches(feature_df_list)

    for batch in storage._data.values():
        # all batches have monotonically increasing numbers (range used to create data)
        assert np.diff(batch["features"][:, 0, 0]).all()  # feature A
        assert np.diff(batch["features"][:, 0, 1]).all()  # feature B


def test_save_and_load_meta():
    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(160)), unit="s"),
                                     "A": range(160),
                                     "B": range(160),
                                     "y": np.ones(160)})
                       for _ in range(1)]

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    batch_generator = Builder(storage,
                              features=["A", "B"],
                              look_back=0, look_forward=0, n_seconds=1, batch_size=16,
                              normalize=False,
                              pseudo_stratify=False)

    batch_generator.generate_and_save_batches(feature_df_list)
    batch_generator.save_meta()

    batch_generator_reload = Builder(storage,
                                     features=["A", "B"],
                                     look_back=99, look_forward=99, n_seconds=99, batch_size=99,
                                     normalize=True,
                                     pseudo_stratify=False)
    batch_generator_reload.load_meta()

    tools.eq_(batch_generator.batch_size, batch_generator_reload.batch_size)
    tools.eq_(batch_generator._features, batch_generator_reload._features)
    tools.eq_(batch_generator._look_forward, batch_generator_reload._look_forward)
    tools.eq_(batch_generator._look_back, batch_generator_reload._look_back)
    tools.eq_(batch_generator._n_seconds, batch_generator_reload._n_seconds)
    tools.eq_(batch_generator._normalize, batch_generator_reload._normalize)
