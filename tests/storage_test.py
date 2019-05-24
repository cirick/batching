from batching.storage import BatchStorageFile
import numpy as np
import nose.tools as tools
from nose import with_setup
import os
import shutil


def setup_func():
    pass


def teardown_func():
    shutil.rmtree("test")


@with_setup(setup_func, teardown_func)
def test_file_storage_directory():
    storage = BatchStorageFile(directory="test")
    tools.eq_(storage.directory, "test")
    assert os.path.exists("test"), True


@with_setup(setup_func, teardown_func)
def test_file_storage_save():
    storage = BatchStorageFile(directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    X_filename, y_filename = storage.save(0, X, y, validation=False)
    assert os.path.isfile(X_filename)
    assert os.path.isfile(y_filename)


@with_setup(setup_func, teardown_func)
def test_file_storage_load():
    storage = BatchStorageFile(directory="test")
    X = np.array([1, 2, 3])
    y = np.array([0, 0, 0])

    X_filename, y_filename = storage.save(0, X, y, validation=False)
    X_data, y_data = storage.load(X_filename, y_filename)
    assert np.array_equal(X_data, X)
    assert np.array_equal(y_data, y)
