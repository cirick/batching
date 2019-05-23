from batching.builder import Builder, BatchMeta
from batching.generator import BatchGenerator

import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG)

feature_set = sorted(["A", "B"])

feature_df_list = [pd.DataFrame({"time": range(1000),
                                 "A": np.random.randn(1000),
                                 "B": np.random.randn(1000),
                                 "y": np.random.randint(2, size=1000)})
                   for i in range(100)]

look_back = 10
look_forward = 10
batch_seconds = 1

batch_meta = BatchMeta()
batch_generator = Builder(batch_meta, feature_set, look_back, look_forward, batch_seconds, batch_size=4096,
                          validation_split=0.5,
                          pseudo_stratify=True, verbose=True, seed=42)
start = time.perf_counter()
batch_generator.generate_and_save_batches(feature_df_list)
print(time.perf_counter() - start)

train_generator = BatchGenerator(batch_meta.train)
val_generator = BatchGenerator(batch_meta.validation)

print([train_generator[i][0].shape for i in range(3)])
print([val_generator[i][0].shape for i in range(3)])
