from batching.builder import Builder
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

batch_generator = Builder(feature_set, look_back, look_forward, batch_seconds, batch_size=4096,
                          pseudo_stratify=True, verbose=True, seed=42)
start = time.perf_counter()
batch_generator.generate_and_save_batches(feature_df_list)
print(time.perf_counter() - start)
