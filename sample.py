from batching.builder import Builder
from batching.storage_meta import StorageMeta
from batching.storage import BatchStorageFile, BatchStorageS3, BatchStorageMemory
from batching.generator import BatchGenerator

import pandas as pd
import numpy as np
import time
import logging
import datetime

logging.basicConfig(level=logging.INFO)

feature_set = sorted(["A", "B"])

look_back = 10
look_forward = 10
batch_seconds = 1
n = 10000
m = 1000

now = int(datetime.datetime.timestamp(datetime.datetime.now()))
feature_df_list = [pd.DataFrame({
    "time": pd.to_datetime(list(range(now, now + (n * batch_seconds), batch_seconds)), unit="s"),
    "A": np.random.randn(n),
    "B": np.random.randn(n),
    "y": np.random.randint(2, size=n)
}) for i in range(m)]

storage_meta = StorageMeta(validation_split=0.5)
storage = BatchStorageMemory(storage_meta)
batch_generator = Builder(storage, feature_set, look_back, look_forward, batch_seconds, batch_size=4096,
                          pseudo_stratify=True, verbose=True, seed=42)
start = time.perf_counter()
batch_generator.generate_and_save_batches(feature_df_list)
print(time.perf_counter() - start)

train_generator = BatchGenerator(storage, is_validation=False)
val_generator = BatchGenerator(storage, is_validation=True)

print([train_generator[i][0].shape for i in range(3)])
print([val_generator[i][0].shape for i in range(3)])
