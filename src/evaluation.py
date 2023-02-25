import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from src.utils import SEED


def get_score_of_classification_model(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                                      y_test: pd.Series) -> float:
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy_score = metrics.accuracy_score(y_test, y_pred) * 100

    return accuracy_score
