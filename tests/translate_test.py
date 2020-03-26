import math
import pandas as pd
import numpy as np
from nose import tools
import datetime
from functools import reduce
from operator import add

from batching.builder import Builder
from batching.storage import BatchStorageMemory
from batching.storage_meta import StorageMeta
from batching.translate import Translate, remove_false_anchors_factory, split_flat_df_by_time_factory


def test_translate_config():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(32)), unit="s"),
                                     "A": np.ones(32),
                                     "B": np.ones(32),
                                     "y": np.ones(32)})
                       for _ in range(1)]

    batch_generator = Builder.memory_builder_factory(feature_set, look_back=3, look_forward=2, batch_size=16,
                                                     batch_seconds=1)
    batch_generator.generate_and_save_batches(feature_df_list)

    tools.eq_(batch_generator.translate._features, list(feature_set))
    tools.eq_(batch_generator.translate.look_forward, 2)
    tools.eq_(batch_generator.translate.look_back, 3)
    tools.eq_(batch_generator.translate._n_seconds, 1)


def test_translate_alone():
    feature_set = sorted(["A", "B"])

    for l in [32, 64, 128]:
        feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(l)), unit="s"),
                                         "A": np.array(list(range(l))),
                                         "B": np.array(list(range(l))),
                                         "y": np.ones(l)})
                           for _ in range(1)]

        for (look_back, look_forward) in [(3, 2), (1, 0)]:
            custom_transforms = list()
            custom_transforms.append(remove_false_anchors_factory("y"))
            custom_transforms.append(split_flat_df_by_time_factory(look_back, look_forward, 1))

            translate = Translate(features=feature_set,
                                  look_back=look_back,
                                  look_forward=look_forward,
                                  n_seconds=1,
                                  custom_transforms=custom_transforms,
                                  normalize=False)

            X, y = translate.scale_and_transform_session(feature_df_list[0])
            tools.eq_(X.shape, (l - (look_back + look_forward), (look_back + look_forward + 1), 2))
            tools.eq_(len(y), l - (look_back + look_forward))

            # first elements should slide forward in time one element at a time
            np.array_equal(X[:, 0, 0], np.array(list(range(l))))
            # second elements should slide forward one at a time starting at 1
            np.array_equal(X[:, 1, 0], np.array(list(range(l))) + 1)


def test_translate_stride():
    feature_set = sorted(["A", "B"])

    for stride in [1, 2, 5]:
        for l in [32, 64, 128]:
            feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(l)), unit="s"),
                                             "A": np.array(list(range(l))),
                                             "B": np.array(list(range(l))),
                                             "y": np.ones(l)})
                               for _ in range(1)]

            for (look_back, look_forward) in [(3, 2), (1, 0)]:
                custom_transforms = list()
                custom_transforms.append(remove_false_anchors_factory("y"))
                custom_transforms.append(split_flat_df_by_time_factory(look_back, look_forward, 1))

                translate = Translate(features=feature_set,
                                      look_back=look_back,
                                      look_forward=look_forward,
                                      n_seconds=1,
                                      stride=stride,
                                      custom_transforms=custom_transforms,
                                      normalize=False)

                X, y = translate.scale_and_transform_session(feature_df_list[0])

                unstrided_len = (l - (look_back + look_forward))
                n_examples_expected = math.ceil(unstrided_len / stride)

                tools.eq_(X.shape, (n_examples_expected, (look_back + look_forward + 1), 2))
                tools.eq_(len(y), n_examples_expected)

                # first elements should slide forward in time one element at a time
                np.array_equal(X[:, 0, 0], np.array(list(range(l))))
                # second elements should slide forward one at a time starting at 1
                np.array_equal(X[:, 1, 0], np.array(list(range(l))) + 1)


def test_remove_anchors():
    y = np.zeros(10)
    y[[1, 5, 7]] = 1
    now = int(datetime.datetime.timestamp(datetime.datetime.now()))
    df = pd.DataFrame({"time": pd.to_datetime(list(range(now, now + 10)), unit="s"),
                       "A": np.ones(10),
                       "B": np.ones(10),
                       "y": y})
    df.index = df["time"]

    assert df["y"].any()
    rfa = remove_false_anchors_factory("y")
    df = rfa(df)
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
    translate = Translate(features=["A", "B"], look_back=0, look_forward=0, n_seconds=1, normalize=True, verbose=True)
    batch_generator = Builder(storage,
                              translate,
                              batch_size=10,
                              pseudo_stratify=False)

    batch_generator.generate_and_save_batches(feature_df_list)

    tools.assert_almost_equal(translate.scaler.mean_[0], 50, delta=1)
    tools.assert_almost_equal(translate.scaler.mean_[1], 150, delta=1)

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
    translate = Translate(features=["A", "B"], look_back=0, look_forward=0, n_seconds=1, normalize=False)
    batch_generator = Builder(storage,
                              translate,
                              batch_size=16,
                              pseudo_stratify=False)

    batch_generator.generate_and_save_batches(feature_df_list)

    for batch in storage._data.values():
        # all batches have monotonically increasing numbers (range used to create data)
        assert np.diff(batch["features"][:, 0, 0]).all()  # feature A
        assert np.diff(batch["features"][:, 0, 1]).all()  # feature B
