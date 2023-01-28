from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

SEED = 42


def get_score_of_classification_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy_score = metrics.accuracy_score(y_test, y_pred) * 100

    return accuracy_score
