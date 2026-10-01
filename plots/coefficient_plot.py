import numpy as np
import matplotlib.pyplot as plt

def coefficient_plot(coef, title="Coefficient Values by Feature Index"):
    plt.figure(figsize=(15, 5))
    
    x = np.arange(1, coef.shape[0] + 1)
    
    mask_zeros = (coef == 0)
    mask_nao_zeros = (coef != 0)
    
    plt.scatter(x[mask_zeros], coef[mask_zeros], color='red', label='Zero')
    plt.scatter(x[mask_nao_zeros], coef[mask_nao_zeros], color='blue', label='Non-zero')
    
    plt.legend()
    
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("Value")
    plt.grid(True)

    plt.show()

    return