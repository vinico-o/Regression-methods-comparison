import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

def generate_data(n_samples, n_features, n_informative, noise, effective_rank, tail_strength):
    X, y, coef = make_regression(
        n_samples = n_samples,        # Define the number of samples
        n_features = n_features,      # Define the number of features
        n_informative = n_informative,     # Define the number of informative features
        noise = noise,             # Define the noise (in standard deviation) level in y
        coef = True,            # Return the coefficients of the underlying linear model
        effective_rank = effective_rank, # Control of correlation between features
        tail_strength = tail_strength, # Control of the decay of the singular values
        random_state = 42       # Set a random seed for reproducibility
    )

    print(f"GENERATED DATA:")
    print(f"Shape of X: {X.shape}")
    print(f"\nX Matrix: \n {X}")

    print(f"Shape of y: {y.shape}")
    print(f"\ny Vector: \n {y}")

    print(f"Shape of coef: {coef.shape}")
    print(f"coef Vector: {coef}")
    print(f"Number of non zero coefs: {np.count_nonzero(coef)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Padronização das features (fit apenas no treino para evitar data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, coef