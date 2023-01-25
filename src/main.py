import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.binner import find_optimal_binning, apply_binning_on_column
from src.data_preprocessing import preprocess_data
from src.test_binning import get_score_of_classification_model, get_score_of_regression_model
from src.utils import load_dataset

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="../data/application_train.csv", type=str,
                        help="Path of the dataset train filepath csv")
    parser.add_argument("--target_column_name", default="TARGET", type=str, help="Column name of the target")
    parser.add_argument("--categorical_column_name", default="ORGANIZATION_TYPE", type=str,
                        help="The categorical feature for applying binning on")
    parser.add_argument("--task", default="classification", type=str, help="Classification or Regression")

    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(args.train_file_path)

    df_columns = df.columns
    nan_columns_summary = df.isnull().sum() != 0
    nan_columns = nan_columns_summary.index[nan_columns_summary].tolist()

    type_to_imputer_startegy = {'float64': 'mean', 'object': 'most_frequent'}

    transformers = []
    for feature in df_columns:
        if feature in nan_columns:
            transformers.append(
                (f'{feature}_imputer', SimpleImputer(strategy=type_to_imputer_startegy[df[feature].dtype.name]),
                 [feature]))
        else:
            transformers.append((f'{feature}_keeper', 'passthrough', [feature]))

    column_trans = ColumnTransformer(transformers)
    df_transformed_data = column_trans.fit_transform(df)
    df_transformed = pd.DataFrame(data=df_transformed_data, columns=df_columns)

    df_train, df_test = train_test_split(df_transformed, test_size=0.25, random_state=SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    x_train, y_train, x_test, y_test = preprocess_data(df_train, df_test, args.target_column_name)

    if args.task == 'classification':
        score_without_binning = get_score_of_classification_model(x_train=x_train, y_train=y_train, x_test=x_test,
                                                                  y_test=y_test)
    else:
        score_without_binning = get_score_of_regression_model(x_train=x_train, y_train=y_train, x_test=x_test,
                                                              y_test=y_test)

    print(f"The score without binning is: {score_without_binning:.3f}")

    optimal_binning = find_optimal_binning(df_train, args.target_column_name, args.categorical_column_name)

    category_values = df_train[args.categorical_column_name].factorize()[1]
    optimal_binning_original = [[category_values[i] for i in b] for b in optimal_binning]

    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    new_col_train = apply_binning_on_column(df_train_copy[args.categorical_column_name], optimal_binning_original)
    df_train_copy[args.categorical_column_name] = new_col_train
    new_col_test = apply_binning_on_column(df_test_copy[args.categorical_column_name], optimal_binning_original)
    df_test_copy[args.categorical_column_name] = new_col_test

    x_train, y_train, x_test, y_test = preprocess_data(df_train_copy, df_test_copy, args.target_column_name)
    with_binning_score = get_score_of_classification_model(x_train, y_train, x_test, y_test)
    print(f"The score with binning is: {with_binning_score:.3f}")


if __name__ == '__main__':
    main()
