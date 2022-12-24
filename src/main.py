import argparse
import time
from src.utils import load_dataset
from src.binner import find_optimal_binning
import warnings

warnings.filterwarnings("ignore")


def _search_for_categorical_feature(dtf_train):
    for col in dtf_train.columns:
        print(col, len(dtf_train[col].value_counts()))


def main():
    train_file_path = 'mushrooms.csv'
    target_column = 'class'
    categorical_feature_column = 'gill-color'

    start = time.time()
    dtf_train = load_dataset(f'../data/{train_file_path}')
    find_optimal_binning(dtf_train, target_column, categorical_feature_column)

    print(f"Took {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    main()
