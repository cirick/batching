import numpy as np


def feature_df_to_nn_input(features, look_back, look_forward, df):
    window_features = []
    x_start = look_back + look_forward
    y_start = look_back
    y_end = len(df["y"]) - look_forward

    for feature in features:
        feature_df = df[feature]
        ts_data = [feature_df.shift(i).values for i in range(look_back + look_forward, -1, -1)]
        window_features.append(np.vstack(ts_data)[:, x_start:])

    # transpose: (n_features, n_seconds, look_back) -> (n_seconds, look_back, n_features)
    return np.stack(window_features).transpose((2, 1, 0)), df.iloc[y_start:y_end]["y"]


def split_flat_df_by_time_gaps(df, gap_seconds, look_back, look_forward):
    gap_idxs = np.where(np.diff(df["time"].values) != np.timedelta64(gap_seconds, 's'))[0].tolist()
    if not gap_idxs:
        return [df]

    start_idx = 0
    valid_sections = []
    for gap_idx in gap_idxs:
        end_idx = gap_idx + 1
        if df.iloc[start_idx:end_idx].shape[0] >= (look_back + look_forward + 1):
            valid_sections.append(df.iloc[start_idx:end_idx])
        start_idx = end_idx
    if df.iloc[start_idx:].shape[0] >= (look_back + look_forward + 1):
        valid_sections.append(df.iloc[start_idx:])

    return valid_sections
