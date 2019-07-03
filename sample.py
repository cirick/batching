from batching.builder import Builder
from batching.generator import BatchGenerator

import numpy as np
import time
import logging

from eight_algos.DataUtils.DfConverter import feature_gen_file_to_df
from concurrent.futures import ThreadPoolExecutor
import pendulum
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDfGenerator():
    def __init__(self, ignore_features=[], filter_presence=None):
        self._ignore_features = ignore_features
        self._filter_presence = filter_presence

    def batch_file_to_df(self, df_path):
        return feature_gen_file_to_df('batch-', df_path, self._ignore_features, self._filter_presence)

    def seconds_file_to_df(self, df_path):
        return feature_gen_file_to_df('seconds-', df_path, self._ignore_features, None)


data_path = ""
csv_path = data_path + "*.csv"

feature_df_generator = FeatureDfGenerator()
logger.info("Load Data")
start = pendulum.now()
with ThreadPoolExecutor() as p:
    batch_sessions_df = p.map(feature_df_generator.batch_file_to_df, glob.glob(csv_path))
logger.info("{}".format(pendulum.now().diff_for_humans(start)))

batch_sessions_df = [df for df in batch_sessions_df if not df.empty]
logger.info(len(batch_sessions_df))

session_ids = [df['session_name'][0] for df in batch_sessions_df]
session_map = {df['session_name'][0]: idx for idx, df in enumerate(batch_sessions_df)}

asleep_sessions_df = list()
for session in batch_sessions_df:
    asleep_mask = np.where(np.logical_and(session["y"] < 4, session["y"] > 0))[0]
    asleep_df = session.iloc[asleep_mask]
    asleep_df.loc[:, "y"] -= 1
    asleep_sessions_df.append(asleep_df)

file_batch_config = {
    "directory": "test",
    "feature_set": sorted([
        'active.log_int',
        'active.range',
        'active.var',
    ]),
    "look_back": 120,
    "look_forward": 60,
    "batch_size": 4096,
    "batch_seconds": 5,
    "validation_split": 0.5,
    "pseudo_stratify": True,
    "stratify_nbatch_groupings": 100,
    "n_workers": None,
    "seed": 42,
    "normalize": True,
    "verbose": True
}
batch_generator = Builder.file_builder_factory(**file_batch_config)

start = time.perf_counter()
batch_generator.generate_and_save_batches(batch_sessions_df)
logger.info(f"Total Duration: {time.perf_counter() - start}")

train_generator = BatchGenerator(batch_generator.storage, is_validation=False)
val_generator = BatchGenerator(batch_generator.storage, is_validation=True)

train_batches = [batch for batch in train_generator]
val_batches = [batch for batch in val_generator]

logger.info(f"num training batches: {len(train_batches)}, num validation batches: {len(val_batches)}")
logger.info(f"training data shapes: {[train_batches[i][0].shape for i in range(3)]}, "
            f"validation data shapes: {[val_batches[i][0].shape for i in range(3)]}")
