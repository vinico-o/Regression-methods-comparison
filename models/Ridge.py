import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

def run_ridge(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)

    for i in alphas:
        Ridge_model = Ridge(alpha=i)
        Ridge_model.fit(X_train, y_train)
        Ridge_y_pred = Ridge_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(Ridge_y_pred, y_test)

        print(f"Number of non zero coefs: {np.count_nonzero(Ridge_model.coef_)}")
        title = f"Ridge Coefficient Values by Feature Index with alpha = {i}"
        coefficient_plot(Ridge_model.coef_, title=title)

def run_ridge_cv(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    RidgeCV_model = RidgeCV()
    RidgeCV_model.fit(X_train, y_train)
    RidgeCV_y_pred = RidgeCV_model.predict(X_test)

    compute_metrics(RidgeCV_y_pred, y_test)

    print(f"Number of non zero coefs: {np.count_nonzero(RidgeCV_model.coef_)}")
    title = f"RidgeCV Coefficient Values by Feature Index with alpha = {RidgeCV_model.alpha_}"
    coefficient_plot(RidgeCV_model.coef_, title=title)

    return RidgeCV_model