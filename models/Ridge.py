import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from plots.metrics import print_coef_info

def run_ridge(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)
    models = []

    for i in alphas:
        Ridge_model = Ridge(alpha=i)
        Ridge_model.fit(X_train, y_train)
        Ridge_y_pred = Ridge_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(Ridge_y_pred, y_test)
        print_coef_info(Ridge_model.coef_)

        title = f"Ridge Coefficient Values by Feature Index with alpha = {i}"
        coefficient_plot(Ridge_model.coef_, title=title)
        models.append(Ridge_model)

    return models

def run_ridge_cv(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    RidgeCV_model = RidgeCV()
    RidgeCV_model.fit(X_train, y_train)
    RidgeCV_y_pred = RidgeCV_model.predict(X_test)

    compute_metrics(RidgeCV_y_pred, y_test)
    print_coef_info(RidgeCV_model.coef_)

    title = f"RidgeCV Coefficient Values by Feature Index with alpha = {RidgeCV_model.alpha_}"
    coefficient_plot(RidgeCV_model.coef_, title=title)

    return RidgeCV_model