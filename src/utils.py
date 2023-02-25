import math
from itertools import combinations
from typing import List, Dict

import pandas as pd

type_to_imputer_strategy = {'float64': 'mean', 'object': 'most_frequent'}
SEED = 42
TEST_SIZE = 0.25
DEFAULT_MIN_UNIQUE = 5
DEFAULT_MAX_UNIQUE = 12


def get_all_possible_partitions(group: List) -> List[List[List[int]]]:
    subsets = [v for a in range(2, len(group)) for v in combinations(group, a)]
    count_splits = math.ceil(len(subsets) / 2)
    possible_splits = []
    for i in range(count_splits):
        possible_splits.append([list(subsets[i]), [e for e in group if e not in subsets[i]]])
    return possible_splits


def get_categorical_columns_by_range_of_uniqueness(df: pd.DataFrame, min_unique: int, max_unique: int,
                                                   target_column: str) -> List[str]:
    df = df.drop(target_column, axis=1)
    df_unique_values = df.nunique()
    df_filtered_unique_values = df_unique_values.ge(min_unique) & df_unique_values.le(max_unique)

    return list(df_filtered_unique_values[df_filtered_unique_values].index)


def load_datasets(dataset_to_target: Dict[str, str]) -> pd.DataFrame:
    read_dataset = lambda x: pd.read_csv(f'../data/{x}')

    datasets_df = pd.DataFrame([], columns=['Name', 'df', 'Target Column'])

    for dataset, target_column in dataset_to_target.items():
        df2 = pd.DataFrame([[dataset, read_dataset(f"{dataset}/{dataset}_dataset.csv"), target_column]],
                           columns=['Name', 'df', 'Target Column'])
        datasets_df = datasets_df.append(df2, ignore_index=True)

    return datasets_df
