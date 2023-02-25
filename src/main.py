import argparse
import time
import warnings
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.binner import apply_binning_on_column, find_optimal_binning
from src.data_preprocessing import preprocess_data, impute_dataframe
from src.evaluation import get_score_of_classification_model
from src.utils import get_categorical_columns_by_range_of_uniqueness, SEED, TEST_SIZE, load_datasets, \
    DEFAULT_MAX_UNIQUE, DEFAULT_MIN_UNIQUE

warnings.simplefilter('ignore')


def _test_single_column(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column_name: str,
                        categorical_column_name: str) -> Tuple[float, List[List[int]], float]:
    start_time = time.time()
    optimal_binning = find_optimal_binning(train_df, target_column_name, categorical_column_name)
    total_time = time.time() - start_time

    df_train_copy = train_df.copy()
    df_test_copy = test_df.copy()
    new_col_train = apply_binning_on_column(df_train_copy[categorical_column_name], optimal_binning)
    df_train_copy[categorical_column_name] = new_col_train
    new_col_test = apply_binning_on_column(df_test_copy[categorical_column_name], optimal_binning)
    df_test_copy[categorical_column_name] = new_col_test

    x_train, y_train, x_test, y_test = preprocess_data(df_train_copy, df_test_copy, target_column_name)
    with_binning_score = get_score_of_classification_model(x_train, y_train, x_test, y_test)

    return with_binning_score, optimal_binning, total_time


def _examine_dataset(df: pd.DataFrame, target_column_name: str, dataset_name: str, min_unique: int, max_unique: int) -> \
        Tuple[Dict, float]:
    df_transformed = impute_dataframe(df=df)

    df_train, df_test = train_test_split(df_transformed, test_size=TEST_SIZE, random_state=SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    x_train, y_train, x_test, y_test = preprocess_data(df_train, df_test, target_column_name)
    score_without_binning = round(get_score_of_classification_model(x_train=x_train, y_train=y_train, x_test=x_test,
                                                                    y_test=y_test), 3)

    categorical_columns = get_categorical_columns_by_range_of_uniqueness(df_train, min_unique=min_unique,
                                                                         max_unique=max_unique,
                                                                         target_column=target_column_name)
    results_summary = {}

    for column in tqdm(categorical_columns, desc=f"{dataset_name} Progress"):
        with_binning_score, optimal_binning, total_time = _test_single_column(df_train, df_test, target_column_name,
                                                                              categorical_column_name=column)
        results_summary[column] = {"score": round(with_binning_score, 3),
                                   "og_unique": sorted(df_train[column].unique()),
                                   "og_n_unique": df_train[column].nunique(),
                                   "new_unique": optimal_binning,
                                   "n_unique": len(optimal_binning),
                                   "total_time": total_time}

    return results_summary, score_without_binning


def get_results(datasets_df: pd.DataFrame, min_unique: int, max_unique: int):
    model_records = []

    for idx, row in tqdm(datasets_df.iterrows(), desc="Datasets Progress", total=len(datasets_df)):
        success_columns, score_without_binning = _examine_dataset(df=row.df, dataset_name=row["Name"],
                                                                  target_column_name=row["Target Column"],
                                                                  min_unique=min_unique, max_unique=max_unique)
        for col in success_columns:
            model_records.append([row["Name"], col, success_columns[col]["score"], score_without_binning,
                                  success_columns[col]["og_unique"], success_columns[col]["og_n_unique"],
                                  success_columns[col]["new_unique"], success_columns[col]["n_unique"],
                                  success_columns[col]["total_time"]])

    return model_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
                        help="Dataset name, stored under dataset_name/dataset_name_dataset.csv")
    parser.add_argument("--target_column_name", type=str, help="Column name of the target")
    parser.add_argument("--min_unique", type=int, default=DEFAULT_MIN_UNIQUE, help="Minimum feature unique values")
    parser.add_argument("--max_unique", type=int, default=DEFAULT_MAX_UNIQUE, help="Maximum feature unique values")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_to_target = {args.dataset_name: args.target_column_name}
    datasets_df = load_datasets(dataset_to_target=dataset_to_target)

    model_records = get_results(datasets_df, min_unique=args.min_unique, max_unique=args.max_unique)
    results = pd.DataFrame(data=model_records,
                           columns=["Dataset", "Column Name", "Optimal Binning Model Score", "Score without Binning",
                                    "og_unique", "og_n_unique", "new_unique", "n_unique", "total_time"])
    results["score_diff"] = results["Optimal Binning Model Score"] - results["Score without Binning"]
    print(results)


if __name__ == '__main__':
    main()
