import numpy as np
from sklearn.linear_model import ElasticNet

def run_elasticnet_alpha(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)

    for i in alphas:
        ElasticNet_model = ElasticNet(alpha=i)
        ElasticNet_model.fit(X_train, y_train)
        ElasticNet_y_pred = ElasticNet_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(ElasticNet_y_pred, y_test)

        print(f"Number of non zero coefs: {np.count_nonzero(ElasticNet_model.coef_)}")
        title = f"ElasticNet Coefficient Values by Feature Index with alpha = {i}"
        coefficient_plot(ElasticNet_model.coef_, title=title)

def run_elasticnet_l1ratio(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)
    l1_ratio = np.array([0, 0.25, 0.50, 0.75, 1])

    for i in l1_ratio:
        ElasticNet_model = ElasticNet(l1_ratio=i)
        ElasticNet_model.fit(X_train, y_train)
        ElasticNet_y_pred = ElasticNet_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(ElasticNet_y_pred, y_test)

        print(f"Number of non zero coefs: {np.count_nonzero(ElasticNet_model.coef_)}")
        title = f"ElasticNet Coefficient Values by Feature Index with l1_ratio = {i}"
        coefficient_plot(ElasticNet_model.coef_, title=title)