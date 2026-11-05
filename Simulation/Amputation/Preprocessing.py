import random
import random as rand
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Union
from scipy import stats
#device = torch.device('cuda')


# Ampute data at a given degree of missingness
'''def ampute_data(dataset, missingness_rate):
    data = dataset.copy()
    sample_idx = 0
    for sample in data:
        for x in range(0, 6, 1):
            if rand.uniform(0, 1) <= missingness_rate:
                data[sample_idx][x] = 0
        sample_idx += 1
    return data'''

def get_mask_matrix(raw_data):
    if type(raw_data) == torch.Tensor:
        mask_matrix = raw_data.cpu().numpy()
    else:
        mask_matrix = raw_data.copy()
    #mask_matrix = np.where(mask_matrix == 0, 1.0, 0.0)
    mask_matrix = np.where(np.isnan(mask_matrix), 1.0, 0.0)
    return mask_matrix

def ampute_data(dataset, missingness_rate):
    data = dataset.copy()
    sample_idx = 0
    for sample in data:
        for x in range(0, 6, 1):
            if rand.random() < missingness_rate:
                data[sample_idx][x] = 0
        sample_idx += 1
    return data

def ampute_single_randomly(dataset, feature, missingness_rate):
    data = dataset
    sample_idx = 0
    for sample in data:
        for x in range(0, feature+1, 1):
            if rand.random() < missingness_rate:
                data[sample_idx][x] = 0
        sample_idx += 1
    return data

def ampute_single(dataset, length):
    data = dataset.copy()
    for sample_idx in range(0, length, 1):
        data[sample_idx][rand.randint(0,5)] = 0
    return data

def get_nan_mask_matrix(raw_data):
    if type(raw_data) == torch.Tensor:
        mask_matrix = raw_data.cpu().numpy()
    else:
        mask_matrix = raw_data.copy()
    mask_matrix = np.where(np.isnan(mask_matrix), 1.0, 0.0)
    return mask_matrix

# Get a list of indexes for missing data (i.e = 0)
def get_missing_indexes(dataset):
    indexes = []
    s = 0
    for sample in dataset:
        s_temp = []
        for x in range(0, 6):
            if sample[x] == 0:
                temp = (s, x)
                s_temp.append(temp)
        if s_temp:
            indexes.append(s_temp)
        s += 1
    return indexes

def get_missing_indexes_nan(dataset):
    indexes = []
    for s, sample in enumerate(dataset):
        s_temp = []
        for x in range(6):  # or use range(len(sample)) if length varies
            if np.isnan(sample[x]):
                s_temp.append((s, x))
        if s_temp:
            indexes.append(s_temp)
    return indexes

def split_indexes_by_feature(indexes):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    s = 0
    for x in indexes:  # For each group of tuples
        for pair in x:  # For an individual tuple in this group
            if pair[1] == 0:  # If this value is 0
                f1.append(pair)  # Append the current pair to f1
            elif pair[1] == 1:
                f2.append(pair)
            elif pair[1] == 2:
                f3.append(pair)
            elif pair[1] == 3:
                f4.append(pair)
            elif pair[1] == 4:
                f5.append(pair)
            elif pair[1] == 5:
                f6.append(pair)
    return f1, f2, f3, f4, f5, f6

# Get data at from dataset given a list of indexes
def get_data_at_indexes(data, indexes):
    list_complete = []
    for x in indexes:
        for s, f in x:
            list_complete.append(data[s][f].item())
    list_complete = np.asarray(list_complete)
    return list_complete

def get_data_at_indexes_by_feature(data, indexes):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    for x in indexes:
        for s, f in x:
            if f == 0:
                f1.append(data[s][f].item())
            elif f == 1:
                f2.append(data[s][f].item())
            elif f == 2:
                f3.append(data[s][f].item())
            elif f == 3:
                f4.append(data[s][f].item())
            elif f == 4:
                f5.append(data[s][f].item())
            elif f == 5:
                f6.append(data[s][f].item())

    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    f3 = np.asarray(f3)
    f4 = np.asarray(f4)
    f5 = np.asarray(f5)
    f6 = np.asarray(f6)

    return f1, f2, f3, f4, f5, f6

def remove_feature(data, feature):
    ys = data[:, feature]
    # ys = np.vstack(ys)
    ys = np.ravel(ys)
    temp = data.copy()
    temp[:, feature] = 0
    xs = temp[:, :]
    return xs, ys

def filter_rows_by_column_np(matrix, column_index):
    matrix = np.asarray(matrix)
    if column_index < 0 or column_index >= 6:
        raise ValueError("Invalid column index")

    return matrix[matrix[:, column_index] != 0]

def split_data_by_zero_rows(data, zero_feature_count=6, held_out_fraction=0.0, seed=None):
    """
    Splits data into two arrays:
    - main_data: 80% of rows where the first `zero_feature_count` features are NOT all 0
    - zero_and_held_out_data: rows where the first `zero_feature_count` features ARE all 0
                              + 20% of the non-zeroed rows

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
    - zero_feature_count: int, number of leading features to check for all-zeros
    - held_out_fraction: float, fraction of non-zeroed rows to hold out
    - seed: int or None, for reproducibility

    Returns:
    - main_data: np.ndarray
    - zero_and_held_out_data: np.ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    # Mask for rows where first N features are all zero
    zero_mask = np.all(data[:, :zero_feature_count] == 0, axis=1)

    zero_rows = data[zero_mask]
    non_zero_rows = data[~zero_mask]

    # Shuffle non-zero rows
    n_non_zero = len(non_zero_rows)
    indices = np.arange(n_non_zero)
    np.random.shuffle(indices)

    split_point = int((1 - held_out_fraction) * n_non_zero)

    main_data = non_zero_rows[indices[:split_point]]
    held_out_data = non_zero_rows[indices[split_point:]]

    # Combine zero-rows and held-out rows
    zero_and_held_out_data = np.vstack([zero_rows, held_out_data])

    return main_data, zero_and_held_out_data

###### Simulating a dataset that is a mix of MAR and MNAR ######

def introduce_missingness_fast(dataset: Union[np.ndarray, pd.DataFrame], missing_proportion: float,
                               mar_fraction: float = 0.6, mar_features: Optional[List[int]] = None,
                               mnar_features: Optional[List[int]] = None, random_state: Optional[int] = None,
                               max_attempts: int = 50) -> Union[np.ndarray, pd.DataFrame]:

    rng = np.random.default_rng(seed=random_state)  # Get a random value
    is_df = isinstance(dataset, pd.DataFrame)
    data = dataset.values.astype(float) if is_df else dataset.astype(float)
    columns = dataset.columns if is_df else None

    n_samples, n_features = data.shape
    total_elements = n_samples * n_features
    n_missing_total = int(missing_proportion * total_elements)
    n_missing_mar = int(mar_fraction * n_missing_total)
    n_missing_mnar = n_missing_total - n_missing_mar

    # Normalize feature indices
    def normalize_features(features):
        if features is None:
            return list(range(n_features))
        if is_df:
            return [columns.get_loc(f) if isinstance(f, str) else f for f in features]
        return features

    if mar_fraction == 0.0:
        mar_features = []
        mnar_features = normalize_features(mnar_features)
    elif mar_fraction == 1.0:
        mar_features = normalize_features(mar_features)
        mnar_features = []
    else:
        mar_features = normalize_features(mar_features)
        mnar_features = normalize_features(mnar_features)

    mar_indices = set()
    mnar_indices = set()

    # -------- Optimized MAR --------
    attempts = 0
    while len(mar_indices) < n_missing_mar and attempts < max_attempts:
        target_feature = rng.choice(mar_features)  # Get target feature
        cond_feature = rng.choice([i for i in range(n_features) if i != target_feature]) # get conditional feature
        values = data[:, cond_feature]  # Get values of the conditional feature

        if np.all(np.isnan(values)):  # If all values in the conditional feature are nan
            attempts += 1  # Increment attempts
            continue  # go to next cycle (aka try again)

        threshold = np.nanpercentile(values, rng.uniform(25, 75))  # get the percentiles across the values, 75% are thresholds
        condition = values > threshold if rng.random() > 0.5 else values < threshold  # condition to make mar
        eligible = np.where(condition & ~np.isnan(data[:, target_feature]))[0]  # check which values are eligible

        if eligible.size > 0:  # if there are any eligible data points
            selected = rng.choice(eligible, size=min(eligible.size, n_missing_mar - len(mar_indices)), replace=False)  # choose which eligible data to make nan
            mar_indices.update((i, target_feature) for i in selected)  # Update indices to be made nan given mar

        attempts += 1  # increment attempts

    # -------- Optimized MNAR --------
    attempts = 0  # reset attempts
    while len(mnar_indices) < n_missing_mnar and attempts < max_attempts:  # conditional statement to ensure attempts < max_attempts
        target_feature = rng.choice(mnar_features)  # choose target feature from options
        values = data[:, target_feature]  # get the data

        if np.all(np.isnan(values)):  # As above
            attempts += 1
            continue

        threshold = np.nanpercentile(values, rng.uniform(25, 75))  # get data percentiles
        condition = values > threshold if rng.random() > 0.5 else values < threshold  # condition for mnar (based on own value)
        eligible = np.where(condition & ~np.isnan(values))[0]  # find eligible data

        if eligible.size > 0:  # if there are any eligible data points
            selected = rng.choice(eligible, size=min(eligible.size, n_missing_mnar - len(mnar_indices)), replace=False)  # select from eligible
            mnar_indices.update((i, target_feature) for i in selected)  # add to indices

        attempts += 1

    # Apply missingness
    for i, j in mar_indices.union(mnar_indices):  # for the indices in mar and mnar indexes
        data[i, j] = np.nan  # make data nan

    return pd.DataFrame(data, columns=columns) if is_df else data  # return the updated data

def introduce_missingness_fast_v2(
    dataset,
    missing_proportion=0.2,
    mar_fraction=0.6,
    mar_features=None,
    mnar_features=None,
    random_state=None,
    max_attempts=50,
    min_observed_frac=0.01,
):

    rng = np.random.default_rng(seed=random_state)
    is_df = isinstance(dataset, pd.DataFrame)
    data = dataset.values.astype(float) if is_df else dataset.astype(float)
    columns = dataset.columns if is_df else None

    n_samples, n_features = data.shape
    print(n_features)
    total_elements = n_samples * n_features
    n_missing_total = int(missing_proportion * total_elements)
    n_missing_mar = int(mar_fraction * n_missing_total)
    n_missing_mnar = n_missing_total - n_missing_mar

    max_missing_per_col = int((1 - min_observed_frac) * n_samples)
    missing_counts = np.zeros(n_features, dtype=int)

    def normalize_features(features):
        if features is None:
            return list(range(n_features))
        if is_df:
            return [columns.get_loc(f) if isinstance(f, str) else f for f in features]
        return features

    if mar_fraction == 0.0:
        mar_features = []
        mnar_features = normalize_features(mnar_features)
    elif mar_fraction == 1.0:
        mar_features = normalize_features(mar_features)
        mnar_features = []
    else:
        mar_features = normalize_features(mar_features)
        mnar_features = normalize_features(mnar_features)

    mar_indices = set()
    mnar_indices = set()

    # -------- MAR --------
    attempts = 0
    while len(mar_indices) < n_missing_mar and attempts < max_attempts:
        target_feature = rng.choice(mar_features)

        if missing_counts[target_feature] >= max_missing_per_col:
            attempts += 1
            continue

        cond_feature = rng.choice([i for i in range(n_features) if i != target_feature])
        values = data[:, cond_feature]

        if np.all(np.isnan(values)):
            attempts += 1
            continue

        threshold = np.nanpercentile(values, rng.uniform(25, 75))
        condition = values > threshold if rng.random() > 0.5 else values < threshold

        eligible = np.where(condition &~np.isnan(data[:, target_feature]))[0]

        if eligible.size > 0:
            selected = rng.choice(
                eligible,
                size=min(eligible.size, n_missing_mar - len(mar_indices)),
                replace=False
            )

            for i in selected:
                if missing_counts[target_feature] < max_missing_per_col:
                    mar_indices.add((i, target_feature))
                    missing_counts[target_feature] += 1

        attempts += 1

    # -------- MNAR --------
    attempts = 0
    while len(mnar_indices) < n_missing_mnar and attempts < max_attempts:
        target_feature = rng.choice(mnar_features)

        if missing_counts[target_feature] >= max_missing_per_col:
            attempts += 1
            continue

        values = data[:, target_feature]

        if np.all(np.isnan(values)):
            attempts += 1
            continue

        threshold = np.nanpercentile(values, rng.uniform(25, 75))
        condition = values > threshold if rng.random() > 0.5 else values < threshold

        eligible = np.where(condition & ~np.isnan(values))[0]

        if eligible.size > 0:
            selected = rng.choice(
                eligible,
                size=min(eligible.size, n_missing_mnar - len(mnar_indices)),
                replace=False
            )

            for i in selected:
                if missing_counts[target_feature] < max_missing_per_col:
                    mnar_indices.add((i, target_feature))
                    missing_counts[target_feature] += 1

        attempts += 1

    # Apply missingness
    for i, j in mar_indices.union(mnar_indices):
        data[i, j] = np.nan

    return pd.DataFrame(data, columns=columns) if is_df else data