from typing import List

import pandas as pd

from src.association import _get_contingency_table
from src.binner import _get_bin_score_categorical_categorical, _get_contingency_table_after_binning
from src.utils import get_all_possible_partitions


def find_optimal_binning_without_frequency(dtf_train: pd.DataFrame, target_column_name: str,
                                           categorical_column_name: str) -> List[List[int]]:
    categorical_column = pd.Series(dtf_train[categorical_column_name].factorize()[0], name=categorical_column_name)
    target_column = pd.Series(dtf_train[target_column_name].factorize()[0], name=target_column_name)

    values_frequencies = categorical_column.value_counts(normalize=True).sort_values()
    original_contingency_table = _get_contingency_table(categorical_column, target_column)

    best_binning = _get_optimal_bins_without_frequency(list(values_frequencies.keys()),
                                                       original_contingency_table)

    category_values = dtf_train[categorical_column_name].factorize()[1]
    best_binning = [[category_values[i] for i in b] for b in best_binning]

    return best_binning


def _get_optimal_bins_without_frequency(categories: List, contingency_table: pd.DataFrame) -> List[List[int]]:
    one_bin_score = _get_bin_score_categorical_categorical([categories], contingency_table)
    max_two_bins_score = 0
    best_bins = None

    for p in get_all_possible_partitions(categories):
        bin_score = _get_bin_score_categorical_categorical(p, contingency_table)
        if bin_score > max_two_bins_score:
            max_two_bins_score = bin_score
            best_bins = p

    if max_two_bins_score <= one_bin_score:
        return [categories]
    else:
        categories = list(contingency_table.index.values)
        return [
            *_get_optimal_bins_without_frequency(best_bins[0],
                                                 _get_contingency_table_after_binning(contingency_table, categories,
                                                                                      best_bins[1],
                                                                                      [best_bins[1]])),
            *_get_optimal_bins_without_frequency(best_bins[1],
                                                 _get_contingency_table_after_binning(contingency_table, categories,
                                                                                      best_bins[0],
                                                                                      [best_bins[0]]))
        ]
