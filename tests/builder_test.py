import pandas as pd
import numpy as np
import boto3
from moto import mock_s3
from nose import tools

from batching.builder import Builder
from batching.storage import BatchStorageMemory
from batching.storage_meta import StorageMeta
from batching.translate import Translate


@tools.raises(Exception)
def test_no_dataset():
    batch_generator = Builder.memory_builder_factory([], 0, 0, 1)
    batch_generator.generate_and_save_batches([])


@mock_s3
def test_builder_config():
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="test_bucket")

    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(32)), unit="s"),
                                     "A": np.ones(32),
                                     "B": np.ones(32),
                                     "y": np.ones(32)})
                       for _ in range(1)]

    batch_generator = Builder.s3_builder_factory(conn.Bucket("test_bucket"),
                                                 feature_set, look_back=2, look_forward=2, batch_size=16,
                                                 batch_seconds=1)
    batch_generator.generate_and_save_batches(feature_df_list)

    tools.eq_(batch_generator.batch_size, 16)
    assert not batch_generator._stratify


def test_builder_storage_meta():
    feature_set = sorted(["A", "B"])

    feature_df_list = [pd.DataFrame({"time": pd.to_datetime(list(range(35)), unit="s"),
                                     "A": np.ones(35),
                                     "B": np.ones(35),
                                     "y": np.ones(35)})
                       for _ in range(1)]

    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    translate = Translate(features=feature_set, look_back=2, look_forward=1, n_seconds=1)
    batch_generator = Builder(storage, translate, batch_size=16)

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

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    translate = Translate(features=feature_set, look_back=2, look_forward=1, n_seconds=1)
    batch_generator = Builder(storage, translate, batch_size=16)

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

    meta = StorageMeta(validation_split=0.5)
    storage = BatchStorageMemory(meta)
    translate = Translate(features=feature_set, look_back=0, look_forward=0, n_seconds=1)
    batch_generator = Builder(storage, translate, batch_size=16,
                              stratify_nbatch_groupings=3,
                              pseudo_stratify=True)

    batch_generator.generate_and_save_batches(feature_df_list)

    assert batch_generator._stratify
    tools.eq_(len(meta.train.ids), 5)
    tools.eq_(len(meta.validation.ids), 5)


def test_save_and_load_meta():
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
    batch_generator.save_meta()

    translate = Translate(features=["A", "B"], look_back=99, look_forward=99, n_seconds=99, normalize=True)
    batch_generator_reload = Builder(storage,
                                     translate,
                                     batch_size=99,
                                     pseudo_stratify=False)
    batch_generator_reload.load_meta()

    tools.eq_(batch_generator.batch_size, batch_generator_reload.batch_size)
    tools.eq_(translate._features, translate._features)
    tools.eq_(translate._look_forward, translate._look_forward)
    tools.eq_(translate._look_back, translate._look_back)
    tools.eq_(translate._n_seconds, translate._n_seconds)
    tools.eq_(translate._normalize, translate._normalize)
