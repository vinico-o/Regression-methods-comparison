from sklearn.linear_model import LinearRegression
import numpy as np
from plots.metrics import print_coef_info

def run_ols(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    OLS_model = LinearRegression()
    OLS_model.fit(X_train, y_train)
    OLS_y_pred = OLS_model.predict(X_test)

    compute_metrics(OLS_y_pred, y_test)
    print_coef_info(OLS_model.coef_)

    title = f"OLS Coefficient Values by Feature Index"
    coefficient_plot(OLS_model.coef_, title=title)

    return OLS_model