import numpy as np
from sklearn.linear_model import Lasso, LassoCV

def run_lasso(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    from sklearn.linear_model import Lasso

    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)

    for i in alphas:
        LASSO_model = Lasso(alpha=i)
        LASSO_model.fit(X_train, y_train)
        LASSO_y_pred = LASSO_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(LASSO_y_pred, y_test)

        print(f"Number of non zero coefs: {np.count_nonzero(LASSO_model.coef_)}")
        title = f"LASSO Coefficient Values by Feature Index with alpha = {i}"
        coefficient_plot(LASSO_model.coef_, title=title)

def run_lasso_cv(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    from sklearn.linear_model import LassoCV

    LASSOCV_model = LassoCV()
    LASSOCV_model.fit(X_train, y_train)
    LASSOCV_y_pred = LASSOCV_model.predict(X_test)

    compute_metrics(LASSOCV_y_pred, y_test)

    print(f"Number of non zero coefs: {np.count_nonzero(LASSOCV_model.coef_)}")
    title = f"LASSO Coefficient Values by Feature Index with alpha = {LASSOCV_model.alpha_}"
    coefficient_plot(LASSOCV_model.coef_, title=title)

    return LASSOCV_model