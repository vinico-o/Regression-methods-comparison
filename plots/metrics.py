import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

def print_coef_info(coef):
    print(f"Number of non zero coefs: {np.count_nonzero(coef)}")
    print(f"Max coef magnitude: {np.abs(coef).max():.4f}")
    print(f"Mean coef magnitude: {np.abs(coef).mean():.4f}")
