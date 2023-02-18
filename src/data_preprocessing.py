import pandas as pd

from typing import Tuple
from pandas import DataFrame, Series
from scipy.sparse.csr import csr_matrix
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def preprocess_data(dtf_train, dtf_test, target_column):
    dtf_train = dtf_train.infer_objects()
    dtf_test = dtf_test.infer_objects()
    x_train = dtf_train.drop(target_column, axis=1)
    x_test = dtf_test.drop(target_column, axis=1)

    y_train = dtf_train[target_column]
    y_test = dtf_test[target_column]

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
