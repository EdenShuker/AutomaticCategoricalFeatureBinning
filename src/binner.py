import math
from typing import Dict, List

import pandas as pd
from pypair.association import categorical_categorical

from .utils import get_all_possible_partitions


def get_bin_score_categorical_categorical(category_to_bin: Dict, binning_column, target_column) -> float:
    new_col_train = binning_column.apply(lambda x: category_to_bin[x] if x in category_to_bin else x)
    bin_score = math.fabs(categorical_categorical(new_col_train, target_column, measure='chisq'))
    return bin_score


def _apply_bin_on_column(column, single_bin, bin_key):
    return column.apply(lambda x: bin_key if x in single_bin else x)


def _get_optimal_bins(rare_categories, binning_column, target_column, bins_keys):
    # we want the bins to be at least with 2 categories. (3 categories can't be split to two bin)
    if len(rare_categories) in [2, 3]:  # TODO: maybe check minimum frequency
        return [rare_categories]

    one_bin_score = get_bin_score_categorical_categorical({c: bins_keys[0] for c in rare_categories}, binning_column,
                                                          target_column)
    max_two_bins_score = 0
    best_bins = None
    for p in get_all_possible_partitions(rare_categories):
        category_to_bin = {}
        for i, subset in enumerate(p):
            for item in subset:
                category_to_bin[item] = bins_keys[i]

        bin_score = get_bin_score_categorical_categorical(category_to_bin, binning_column, target_column)
        if bin_score > max_two_bins_score:
            max_two_bins_score = bin_score
            best_bins = p

    if max_two_bins_score <= one_bin_score:
        return [rare_categories]

    else:
        return [
            *_get_optimal_bins(best_bins[0], _apply_bin_on_column(binning_column, best_bins[1], bins_keys[0]),
                               target_column, bins_keys[1:]),
            *_get_optimal_bins(best_bins[1], _apply_bin_on_column(binning_column, best_bins[0], bins_keys[0]),
                               target_column, bins_keys[1:])
        ]


def get_category_to_bin(bins: List[List], bins_keys: List) -> Dict:
    category_to_bin = {}
    for i, b in enumerate(bins):
        for c in b:
            category_to_bin[c] = bins_keys[i]

    return category_to_bin


def find_optimal_binning(dtf_train: pd.DataFrame, target_column_name: str, categorical_column_name: str):
    """
    supports only nominal variables.
    """
    # TODO: Right now support only nominal features and nominal target values
    categorical_column = pd.Series(dtf_train[categorical_column_name].factorize()[0],
                                   name=categorical_column_name).apply(str)
    target_column = pd.Series(dtf_train[target_column_name].factorize()[0], name=target_column_name).apply(str)

    value_frequencies = categorical_column.value_counts(normalize=True).sort_values()
    categories = list(value_frequencies.keys())
    num_categories = len(categories)
    bins_keys = list(map(str, range(num_categories, num_categories * 2)))
    initial_bin_score = get_bin_score_categorical_categorical({c: bins_keys[i] for i, c in enumerate(categories)},
                                                              categorical_column, target_column)
    print(f"Initial bin score = {initial_bin_score:.2f}")

    # TODO: better statistic on what range to check
    # TODO: maybe wa want maximum number of categories for performances
    max_frequency = 0.05

    max_binning_score = 0
    best_binning = None
    for i, (category, frequency) in enumerate(list(value_frequencies.items())[1:]):
        if frequency > max_frequency:
            break
        rare_categories = value_frequencies[value_frequencies <= frequency].keys()
        bins = _get_optimal_bins(list(rare_categories), categorical_column, target_column, bins_keys)
        bin_score = get_bin_score_categorical_categorical(get_category_to_bin(bins, bins_keys), categorical_column,
                                                          target_column)
        print(f"frequency={frequency:.2f}, bin_score={bin_score:.2f}, bins={bins}")
        if bin_score > max_binning_score:
            max_binning_score = bin_score
            best_binning = bins

    print(f"Best binning: {best_binning}")
    print(f"Binning score: {max_binning_score}")
