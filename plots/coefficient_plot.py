import numpy as np
import matplotlib.pyplot as plt

def coefficient_plot(coef, title="Coefficient Values by Feature Index"):
    plt.figure(figsize=(15, 5))
    
    x = np.arange(1, coef.shape[0] + 1)
    plt.scatter(x, coef)

    
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("Value")
    plt.grid(True)

    plt.show()

    return