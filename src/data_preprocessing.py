from typing import Tuple

import pandas as pd
from pandas import DataFrame, Series
from scipy.sparse.csr import csr_matrix
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from src.utils import type_to_imputer_strategy


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = train_df.infer_objects()
    test_df = test_df.infer_objects()
    x_train = train_df.drop(target_column, axis=1)
    x_test = test_df.drop(target_column, axis=1)

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    x_train, y_train, x_test, y_test = encode_categorical_data(x_train=x_train, y_train=y_train, x_test=x_test,
                                                               y_test=y_test)

    return x_train, y_train, x_test, y_test


def encode_categorical_data(x_train: DataFrame, y_train: Series, x_test: DataFrame, y_test: Series) -> Tuple[
    DataFrame, Series, DataFrame, Series]:
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()

    transformer = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categorical_features),
                                          remainder='passthrough')

    x_train_transformed = transformer.fit_transform(x_train)
    if isinstance(x_train_transformed, csr_matrix):
        x_train_transformed = x_train_transformed.toarray()
    x_train_transformed = pd.DataFrame(x_train_transformed, columns=transformer.get_feature_names_out())

    x_test_transformed = transformer.transform(x_test)
    if isinstance(x_test_transformed, csr_matrix):
        x_test_transformed = x_test_transformed.toarray()
    x_test_transformed = pd.DataFrame(x_test_transformed, columns=transformer.get_feature_names_out())

    if y_train.dtype.name == 'object':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    return x_train_transformed, y_train, x_test_transformed, y_test


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_columns = df.columns
    nan_columns_summary = df.isnull().sum() != 0
    nan_columns = nan_columns_summary.index[nan_columns_summary].tolist()

    transformers = []
    for feature in df_columns:
        if feature in nan_columns:
            transformers.append(
                (f'{feature}_imputer', SimpleImputer(strategy=type_to_imputer_strategy[df[feature].dtype.name]),
                 [feature]))
        else:
            transformers.append((f'{feature}_keeper', 'passthrough', [feature]))

    column_trans = ColumnTransformer(transformers)
    df_transformed_data = column_trans.fit_transform(df)
    transformed_df = pd.DataFrame(data=df_transformed_data, columns=df_columns)

    return transformed_df
