from batching.translate import Translate, split_flat_df_by_time_factory
import pandas as pd
import numpy as np
import nose.tools as tools


def test_feature_sequencing():
    n = 50
    features = ["A", "B"]
    df = pd.DataFrame({"time": range(n),
                       "A": np.random.randn(n),
                       "B": np.random.randn(n),
                       "y": np.random.randint(2, size=n)})

    for i in range(10):
        lookback = i
        lookforward = i

        translate = Translate(features, lookback, lookforward)
        X_res, y_res = translate._feature_df_to_nn_input(df)
        tools.eq_(X_res.shape, (n - (lookback + lookforward), lookback + lookforward + 1, len(features)))


def test_time_gaps():
    n = 50
    for gap in range(1, 3):
        df = pd.DataFrame({"time": pd.to_datetime(list(range(0, n * gap, gap)), unit="s"),
                           "A": np.random.randn(n),
                           "B": np.random.randn(n),
                           "y": np.random.randint(2, size=n)})

        sfd = split_flat_df_by_time_factory(look_back=5, look_forward=5, n_seconds=gap)
        res = sfd(df)
        assert df.equals(res[0])
        tools.eq_(len(res), 1)


def test_time_gaps_split():
    n = 50
    df = pd.DataFrame({"time": range(n),
                       "A": np.random.randn(n),
                       "B": np.random.randn(n),
                       "y": np.random.randint(2, size=n)})

    df.loc[25:, "time"] += 1
    df["time"] = pd.to_datetime(df["time"], unit="s")

    sfd = split_flat_df_by_time_factory(look_back=5, look_forward=5, n_seconds=1)
    res = sfd(df)
    tools.eq_(len(res), 2)
    assert df.equals(pd.concat(res, axis=0))


def test_time_gaps_too_small_segments():
    n = 50
    gap = 2
    df = pd.DataFrame({"time": pd.to_datetime(list(range(0, n * gap, gap)), unit="s"),
                       "A": np.random.randn(n),
                       "B": np.random.randn(n),
                       "y": np.random.randint(2, size=n)})

    sfd = split_flat_df_by_time_factory(look_back=5, look_forward=5, n_seconds=1)
    res = sfd(df)
    tools.eq_(len(res), 0)


def test_time_gaps_fully_segmented():
    n = 50
    gap = 2
    df = pd.DataFrame({"time": pd.to_datetime(list(range(0, n * gap, gap)), unit="s"),
                       "A": np.random.randn(n),
                       "B": np.random.randn(n),
                       "y": np.random.randint(2, size=n)})

    sfd = split_flat_df_by_time_factory(look_back=0, look_forward=0, n_seconds=1)
    res = sfd(df)
    tools.eq_(len(res), n)
