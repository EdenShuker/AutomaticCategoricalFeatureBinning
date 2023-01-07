from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_score_of_regression_model(dtf_train, dtf_test, target_column):
    # separate X from y
    x_train = dtf_train.drop(target_column, axis=1)
    x_test = dtf_test.drop(target_column, axis=1)

    y_train = dtf_train[target_column]
    y_test = dtf_test[target_column]

    model = LinearRegression()
    prediction = model.fit(x_train, y_train).predict(x_test)

    # TODO: don't sure this is the score we want
    return r2_score(y_test, prediction)
