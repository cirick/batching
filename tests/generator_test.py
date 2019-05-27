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
    train_generator = BatchGenerator(storage, is_validation=False)

    X, y = train_generator[0]
    tools.eq_(X.shape, (16, 5, 2))
    assert np.array_equal(X, np.zeros(X.shape))
