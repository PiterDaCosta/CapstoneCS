import numpy as np
import pandas as pd

def std_and_mean(data):
    X_avgs = np.mean(data, axis=0)

    # ddof = 1 (Degrees of freedom)
    # Give us the unbiased std_dev
    X_std = np.std(data, axis=0, ddof=1)

    print(X_std)
    print(X_std)

def numpy_basics(data):
    # Transpose matrix
    transposed = data.T

    # Inverse
    A = np.array([[6, 1, 1, 3],
                  [4, -2, 5, 1],
                  [2, 8, 7, 6],
                  [3, 1, 9, 7]])
    A_inv = np.linalg.inv(A)

# TODO:
# * Create histograms
# * Calculate quantiles of a dist
# * Do a test (t-test?)

def load_data():
    df = pd.read_csv('data/mortality.csv')
    return df

if __name__ == '__main__':
    data = load_data()
    std_and_mean(data)
    numpy_basics(data)
