# !/usr/bin/env python3

# Import arbitrary dataset
from sklearn.datasets import load_diabetes
import numpy as np

# Calculate LS objective gradient
def calc_gradient(x, A, b):
    return A.T @ ((A @ x) - b)

if __name__ == "__main__":
    step_size = 0.01
    num_iterations = 200000
    
    A, b = load_diabetes(return_X_y=True)
    x_0 = np.zeros(A.shape[1])

    # Loop through gradient descent
    x_t = x_0
    for t in range(num_iterations):
        x_t_plus_1 = x_t - step_size * calc_gradient(x_t, A, b)
        x_t = x_t_plus_1
    
    # Print error  
    x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"Error = {np.linalg.norm(x_t - x_hat)}")