import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from plots.metrics import print_coef_info

def run_elasticnet_alpha(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)
    models = []

    for i in alphas:
        ElasticNet_model = ElasticNet(alpha=i)
        ElasticNet_model.fit(X_train, y_train)
        ElasticNet_y_pred = ElasticNet_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(ElasticNet_y_pred, y_test)
        print_coef_info(ElasticNet_model.coef_)

        title = f"ElasticNet Coefficient Values by Feature Index with alpha = {i}"
        coefficient_plot(ElasticNet_model.coef_, title=title)
        models.append(ElasticNet_model)

    return models

def run_elasticnet_l1ratio(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # defining alphas as 0.001 ... 100
    alphas = np.logspace(-3, 2, num=6)
    l1_ratio = np.array([0, 0.25, 0.50, 0.75, 1])
    models = []

    for i in l1_ratio:
        ElasticNet_model = ElasticNet(l1_ratio=i)
        ElasticNet_model.fit(X_train, y_train)
        ElasticNet_y_pred = ElasticNet_model.predict(X_test)

        print(f"Metrics for i = {i}:")
        compute_metrics(ElasticNet_y_pred, y_test)
        print_coef_info(ElasticNet_model.coef_)

        title = f"ElasticNet Coefficient Values by Feature Index with l1_ratio = {i}"
        coefficient_plot(ElasticNet_model.coef_, title=title)
        models.append(ElasticNet_model)

    return models

def run_elasticnet_cv(X_train, X_test, y_train, y_test, compute_metrics, coefficient_plot):
    # The ElasticNet model by default uses the l1_ratio = 0.5 parameter.
    # So, we can define a vector of l1 ratios to check the best one.
    l1_ratios = np.arange(0.1, 1.1, 0.1)

    ElasticNetCV_model = ElasticNetCV(l1_ratio=l1_ratios, random_state=42)
    ElasticNetCV_model.fit(X_train, y_train)
    ElasticNetCV_y_pred = ElasticNetCV_model.predict(X_test)

    compute_metrics(ElasticNetCV_y_pred, y_test)
    print_coef_info(ElasticNetCV_model.coef_)

    title = f"ElasticNetCV Coefficient Values by Feature Index with alpha = {ElasticNetCV_model.alpha_:.2f} and l1_ratio = {ElasticNetCV_model.l1_ratio_:.2f}"
    coefficient_plot(ElasticNetCV_model.coef_, title=title)

    return ElasticNetCV_model