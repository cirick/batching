from batching.storage import BatchStorageFile, BatchStorageMemory
import numpy as np
import nose.tools as tools
from nose import with_setup
import os
import shutil

from batching.storage_meta import StorageMeta


def setup_func():
    pass


def teardown_func():
    shutil.rmtree("test")


def test_mem_storage_save():
    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    filename = storage.save(X, y)
    assert filename in list(storage._data.keys())


def test_mem_storage_load():
    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    X_data, y_data = storage.load(0)
    assert np.array_equal(X_data, X)
    assert np.array_equal(y_data, y)


def test_mem_storage_metadata():
    meta = StorageMeta()
    storage = BatchStorageMemory(meta)
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    storage.save_meta({})
    params = storage.load_meta()
    assert len(params["train_ids"]) == 1
    assert params["train_map"][params["train_ids"][0]] == "ID_0"
    assert len(params["val_ids"]) == 0


def test_mem_storage_metadata_val():
    meta = StorageMeta(validation_split=1.0)
    storage = BatchStorageMemory(meta)
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    storage.save_meta({})
    params = storage.load_meta()
    assert len(params["val_ids"]) == 1
    assert params["val_map"][params["val_ids"][0]] == "IDv_0"
    assert len(params["train_ids"]) == 0


@with_setup(setup_func, teardown_func)
def test_file_storage_directory():
    meta = StorageMeta()
    storage = BatchStorageFile(meta, directory="test")
    tools.eq_(storage.directory, "test")
    assert os.path.exists("test"), True


@with_setup(setup_func, teardown_func)
def test_file_storage_save():
    meta = StorageMeta()
    storage = BatchStorageFile(meta, directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    filename = storage.save(X, y)
    assert os.path.isfile(filename)


@with_setup(setup_func, teardown_func)
def test_file_storage_load():
    meta = StorageMeta()
    storage = BatchStorageFile(meta, directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    X_data, y_data = storage.load(0)
    assert np.array_equal(X_data, X)
    assert np.array_equal(y_data, y)


@with_setup(setup_func, teardown_func)
def test_file_storage_metadata():
    meta = StorageMeta()
    storage = BatchStorageFile(meta, directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    storage.save_meta({})
    params = storage.load_meta()
    assert len(params["train_ids"]) == 1
    assert params["train_map"][params["train_ids"][0]] == "ID_0"
    assert len(params["val_ids"]) == 0


@with_setup(setup_func, teardown_func)
def test_file_storage_metadata_val():
    meta = StorageMeta(validation_split=1.0)
    storage = BatchStorageFile(meta, directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    storage.save(X, y)
    storage.save_meta({})
    params = storage.load_meta()
    assert len(params["val_ids"]) == 1
    assert params["val_map"][params["val_ids"][0]] == "IDv_0"
    assert len(params["train_ids"]) == 0
