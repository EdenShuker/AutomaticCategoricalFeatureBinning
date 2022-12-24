import math
import sys
from itertools import permutations, combinations, chain

import pandas as pd
import logging

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


def get_all_possible_partitions(group):
    subsets = [v for a in range(2, len(group)) for v in combinations(group, a)]
    count_splits = math.ceil(len(subsets) / 2)
    possible_splits = []
    for i in range(count_splits):
        possible_splits.append([list(subsets[i]), [e for e in group if e not in subsets[i]]])
    return possible_splits


def get_all_possible_splits_by_order(group):
    if len(group) == 1:
        return [[group]]
    possible_splits = [[group]]
    for i in range(1, len(group)):
        subset = group[:i]
        next_subsets = get_all_possible_splits_by_order(group[i:])
        for n in next_subsets:
            possible_splits.append([subset, *n])
    return possible_splits


def get_all_possible_bins(group):
    bins_options = []
    for perm in permutations(group):
        possible_bins = get_all_possible_splits_by_order(perm)
        bins_options.extend(possible_bins)
    print(len(bins_options))
    return bins_options
