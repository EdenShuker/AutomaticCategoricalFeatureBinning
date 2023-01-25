import logging
import math
import sys
from itertools import combinations

import pandas as pd

logger = logging.getLogger("categorical binner")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_dataset(train_dataset_file: str) -> pd.DataFrame:
    try:
        data_frame = pd.read_csv(train_dataset_file, index_col='Id')
    except Exception:
        data_frame = pd.read_csv(train_dataset_file)
    return data_frame


def get_all_possible_partitions_without_frequency(group):
    subsets = [v for a in range(len(group)) for v in combinations(group, a)]
    count_splits = math.ceil(len(subsets) / 2)
    possible_splits = []
    for i in range(count_splits):
        possible_splits.append([list(subsets[i]), [e for e in group if e not in subsets[i]]])
    return possible_splits


def get_all_possible_partitions(group):
    subsets = [v for a in range(2, len(group)) for v in combinations(group, a)]
    count_splits = math.ceil(len(subsets) / 2)
    possible_splits = []
    for i in range(count_splits):
        possible_splits.append([list(subsets[i]), [e for e in group if e not in subsets[i]]])
    return possible_splits


def _search_for_categorical_feature(dtf_train):
    for col in dtf_train.columns:
        print(col, len(dtf_train[col].value_counts()))


def get_all_possible_splits(possible_values):
    b = [list(c) for i in range(len(possible_values)) for c in combinations(possible_values, i + 1)]
    check = []
    for elem in b:
        remainder = [x for x in possible_values if x not in elem]
        if remainder not in check and remainder:
            check.append([elem, remainder])

    return check
