import argparse
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils import load_dataset
from src.binner import find_optimal_binning, apply_binning_on_column
from src.test_binning import get_score_of_regression_model
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_filepath", type=str, help="Path of the dataset train filepath csv")
    parser.add_argument("target", type=str, help="Column name of the target")
    parser.add_argument("feature", type=str, help="The categorical feature for applying binning on")

    return parser.parse_args()


def main():
    start = time.time()
    # args = parse_args()
    target_column_name = "class"
    categorical_column_name = "gill-color"
    train_filepath = Path(__file__).parent.parent.parent / "data" / "mushrooms.csv"
    dtf = load_dataset(str(train_filepath))
    dtf_train, dtf_test = train_test_split(dtf, test_size=0.25)
    optimal_bins = find_optimal_binning(dtf_train, target_column_name=target_column_name,
                                        categorical_column_name=categorical_column_name)
    print(f"Took {time.time() - start:.2f} seconds")
    without_binning_score = get_score_of_regression_model(dtf_train, dtf_test, target_column_name)
    new_col_train = apply_binning_on_column(dtf_train[categorical_column_name], optimal_bins)
    dtf_train[categorical_column_name] = new_col_train
    new_col_test = apply_binning_on_column(dtf_test[categorical_column_name], optimal_bins)
    dtf_test[categorical_column_name] = new_col_test
    with_binning_score = get_score_of_regression_model(dtf_train, dtf_test, target_column_name)
    print(f"Before binning score {with_binning_score:.3f}, After binning score {with_binning_score:.3f}")


if __name__ == '__main__':
    main()
