import numpy as np
import pandas as pd
import scipy as scipy


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

    # Matrix multiplication
    mult = np.matmul(A, A_inv)
    mult = A @ A_inv


def tests():
    # TODO: Complete...
    pass



def distributions():
    dist = scipy.stats.chi2(df=5)

    mean = dist.mean()
    median = dist.median()
    std = dist.std()
    var = dist.var()

    # CDF of a value
    c = dist.cdf(x=2)

    # PDF of a value
    c = dist.pdf(x=2)

    # Expectancy of functions of the variable
    f = lambda x: (x ** 2 - 1) / 2
    expectancy_of_f = dist.expect(f)

    # Generating random values from a normal distribution
    n_mu = 0
    n_std = 1
    x = scipy.stats.norm.rvs(n_mu, n_std, size=1000)

    # Fitting a distribution given a set of values.
    mu_fit, std_fit = scipy.stats.norm.fit(x)
    print(mu_fit)
    print(std_fit)


def pandas_filtering():
    data = pd.DataFrame(
        {
         'Last_Name': ['Smith', None, 'Brown'],
         'First_Name': ['John', 'Mike', 'Bill'],
         'Age': [35, 45, None]
        }
    )

    # Sort by one column values
    data = data.sort_values(by='Age', ascending=False)

    # Filter all rows with a particular condition
    data = data[data.Age > 36]

    # Select subset of columns
    data = data[['Age', 'Last_Name']]

    # Drop a column
    data = data.drop(['Age'], axis=1)

    # Filtering all rows with null values in any of its columns
    data = data.dropna()



# TODO:
# * Create histograms
# * Do a test (t-test?)
# * statsmodels + scikit-learn + numpy + pandas


def load_data():
    df = pd.read_csv('data/mortality.csv')
    return df


if __name__ == '__main__':
    data = load_data()
    pandas_filtering()
    std_and_mean(data)
    numpy_basics(data)
    distributions()
    tests()

