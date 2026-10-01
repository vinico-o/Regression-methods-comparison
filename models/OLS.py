from sklearn.linear_model import LinearRegression
import numpy as np

def run_ols(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    OLS_model = LinearRegression()
    OLS_model.fit(X_train, y_train)
    OLS_y_pred = OLS_model.predict(X_test)

    compute_metrics(OLS_y_pred, y_test)

    print(f"Number of non zero coefs: {np.count_nonzero(OLS_model.coef_)}")
    title = f"OLS Coefficient Values by Feature Index"
    coefficient_plot(OLS_model.coef_, title=title)

    return OLS_model