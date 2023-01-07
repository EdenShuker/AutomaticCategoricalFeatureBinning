import pandas as pd
from typing import Dict, List
from .association import _get_contingency_table, uncertainty_coefficient
from more_itertools import flatten

from .utils import get_all_possible_partitions


def apply_binning_on_column(binning_column, bins):
    category_to_bin = {c: b[0] for b in bins for c in b}
    return binning_column.apply(lambda x: category_to_bin[x] if x in category_to_bin else x)


def _get_optimal_bins(rare_categories, contingency_table):
    # we want the bins to be at least with 2 categories. (3 categories can't be split to two bin)
    if 2 <= len(rare_categories) <= 3:  # TODO: maybe check minimum frequency
        return [rare_categories]

    one_bin_score = _get_bin_score_categorical_categorical([rare_categories], contingency_table)
    max_two_bins_score = 0
    best_bins = None
    for p in get_all_possible_partitions(rare_categories):
        bin_score = _get_bin_score_categorical_categorical(p, contingency_table)
        if bin_score > max_two_bins_score:
            max_two_bins_score = bin_score
            best_bins = p

    if max_two_bins_score <= one_bin_score:
        return [rare_categories]
    else:
        categories = list(contingency_table.index.values)
        return [
            *_get_optimal_bins(best_bins[0],
                               _get_contingency_table_after_binning(contingency_table, categories, best_bins[1],
                                                                    [best_bins[1]])),
            *_get_optimal_bins(best_bins[1],
                               _get_contingency_table_after_binning(contingency_table, categories, best_bins[0],
                                                                    [best_bins[0]]))
        ]


def _get_contingency_table_after_binning(original_table: pd.DataFrame, categories: List, rare_categories: List,
                                         bins: List[List[int]]):
    to_keep_categories = list(set(categories) - set(rare_categories))
    new_table = original_table.loc[to_keep_categories]
    for b in bins:
        new_row = original_table.loc[b].sum()
        new_row.name = b[0]
        new_table = new_table.append(new_row)

    return new_table


def _get_bin_score_categorical_categorical(bins, original_contingency_table: pd.DataFrame) -> float:
    categories = list(original_contingency_table.index.values)
    rare_categories = list(flatten(bins))
    contingency_table = _get_contingency_table_after_binning(original_contingency_table, categories, rare_categories,
                                                             bins)
    bin_score = uncertainty_coefficient(contingency_table.values)
    return bin_score


def find_optimal_binning(dtf_train: pd.DataFrame, target_column_name: str, categorical_column_name: str):
    """
    supports only nominal variables.
    """
    # TODO: Right now support only nominal features and nominal target values
    categorical_column = pd.Series(dtf_train[categorical_column_name].factorize()[0],
                                   name=categorical_column_name)
    target_column = pd.Series(dtf_train[target_column_name].factorize()[0], name=target_column_name)

    values_frequencies = categorical_column.value_counts(normalize=True).sort_values()
    original_contingency_table = _get_contingency_table(categorical_column, target_column)

    initial_bin_score = _get_bin_score_categorical_categorical([], original_contingency_table)
    print(f"Initial bin score = {initial_bin_score:.3f}")

    # TODO: better statistic on what range to check
    # TODO: maybe wa want maximum number of categories for performances
    max_frequency = 0.1

    max_binning_score = 0
    best_binning = []
    for i, (category, frequency) in enumerate(list(values_frequencies.items())[1:]):
        if frequency > max_frequency:
            break
        rare_categories = values_frequencies[values_frequencies <= frequency].keys()
        bins = _get_optimal_bins(list(rare_categories), contingency_table=original_contingency_table)
        bin_score = _get_bin_score_categorical_categorical(bins, original_contingency_table)
        print(f"frequency={frequency:.3f}, bin_score={bin_score:.3f}, bins={bins}")
        if bin_score > max_binning_score:
            max_binning_score = bin_score
            best_binning = bins

    print(f"Best binning: {best_binning}")
    print(f"Binning score: {max_binning_score}")
    return best_binning
