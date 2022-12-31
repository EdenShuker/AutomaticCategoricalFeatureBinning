import argparse
import time
from pathlib import Path
from src.utils import load_dataset
from src.binner import find_optimal_binning
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_filepath", type=str, help="Path of the dataset train filepath csv")
    parser.add_argument("target", type=str, help="Column name of the target")
    parser.add_argument("feature", type=str, help="The categorical feature for applying binning on")

    return parser.parse_args()


def _search_for_categorical_feature(dtf_train):
    for col in dtf_train.columns:
        print(col, len(dtf_train[col].value_counts()))


def main():
    start = time.time()
    # args = parse_args()
    train_filepath = Path(__file__).parent.parent.parent / "data" / "mushrooms.csv"
    dtf_train = load_dataset(str(train_filepath))
    print(len(dtf_train))
    # print(dtf_train["birth_state"].value_counts(normalize=True))
    # find_optimal_binning(dtf_train, "party_cd", "birth_state")
    find_optimal_binning(dtf_train, "class", "gill-color")

    print(f"Took {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    main()
