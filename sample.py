from batching.builder import Builder
from batching.generator import BatchGenerator

import pandas as pd
import numpy as np
import time
import logging
import uuid

from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data config
# 10k features + y per session
n = 10000
# Timestep spacing in seconds for features
timesteps_seconds = 5

# Generate a sample dataframe with 2 features and n timesteps
now = datetime.utcnow().replace(microsecond=0)
ts = pd.to_datetime([now + timedelta(seconds=i * timesteps_seconds) for i in range(n)])
X = np.sin(np.linspace(1, n + 1, n)) + np.random.normal(scale=0.1, size=n)
y = np.random.randint(0, 2, n)
session = pd.DataFrame({"feat1": X, "feat2": X, "feat3": X, "feat4": X, "y": y, "time": ts})
session["session_id"] = str(uuid.uuid4())

# Dataset consists of a certain number of sessions (sample 10)
dataset = [session for _ in range(9)]

# filter sessions with certain ID
session_filt = session.copy()
session_filt["session_id"] = 0
dataset += [session_filt]


def session_filter(session):
    session_id = session["session_id"].iloc[0]
    filter_session = session_id == 0
    if filter_session:
        logger.info(f"Filtering session {session_id}")
    return filter_session


# Configuration for building batches
file_batch_config = {
    "directory": "cache",  # output directory
    "feature_set": sorted([
        'feat1',
        'feat2',
        'feat3',
        'feat4'
    ]),
    "look_back": 30,  # sequence model / RNN timesteps looking back
    "look_forward": 30,  # sequence model / RNN timesteps looking forward (total window = look_back + look_forward + 1)
    "batch_size": 1024,  # size of training/val batches
    "stride": 2,
    "batch_seconds": timesteps_seconds,  # timestep size in seconds
    "validation_split": 0.5,  # train/test split
    "pseudo_stratify": True,  # stratify batches (done streaming so pseudo-stratification)
    "stratify_nbatch_groupings": 10,  # number of batches to look at for stratification ratios
    "n_workers": None,  # n_workers for ProcessPoolExecutor. None means ProcessPoolExecutor(n_workers=None) / default
    "seed": 42,  # random seed for repeatability
    "normalize": True,  # use StandardScaler to normalize features
    "session_norm_filter": session_filter,
    "verbose": True  # debug logs
}

# Create builder for saving to files
batch_generator = Builder.file_builder_factory(**file_batch_config)

# Generate batches
start = time.perf_counter()
batch_generator.generate_and_save_batches(dataset)
logger.info(f"Total Duration: {time.perf_counter() - start}")

# Train and validation generators that can be passed to tf/keras fit_generator
train_generator = BatchGenerator(batch_generator.storage, is_validation=False)
val_generator = BatchGenerator(batch_generator.storage, is_validation=True)

# Consume in sample code for stats
train_batches = list(train_generator)
val_batches = list(val_generator)

logger.info(f"num training batches: {len(train_batches)}, num validation batches: {len(val_batches)}")
logger.info(f"training data shapes: {[train_batches[i][0].shape for i in range(3)]}, "
            f"validation data shapes: {[val_batches[i][0].shape for i in range(3)]}")
logger.info(f"Batches location: ./{file_batch_config.get('directory')}")
