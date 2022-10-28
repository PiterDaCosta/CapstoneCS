import numpy as np
import pandas as pd
import scipy as scipy
import matplotlib
import matplotlib.pyplot


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
    # Example: chi squared distribution with 5 degrees of freedom.
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
    # get dataframe statistics
    print(data.describe())

    # Integer location indexing of elements
    i = 2  # row
    j = 1  # column
    d = data.iloc[i, j]

    # Second row (indices starts at 0)
    r = data.iloc[1]

    # Second column
    col = data.iloc[:, 1]

    # Label indexing
    # All rows, just "Last_Name" and "Age" columns
    selection = data.loc[:, ['Last_Name', 'Age']]

    # Count NaN values under a single DataFrame column
    sum = data.isna().sum()

    # Count NaN values under an entire DataFrame
    sum = data.isna().sum().sum()

    # Count NaN values across a single DataFrame row
    row_index = 1
    sum = data.loc[row_index].isna().sum().sum()

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

    return

def charts(data):
    # Basic plotting with matplotlib
    X = np.linspace(0, 2*np.pi, 180)
    Y = np.cos(X)
    matplotlib.pyplot.plot(X, Y, 'r-') # 'r' color (r, b, ...) , '-' marker (., -, o, ...)
    matplotlib.pyplot.show()

    # Plot with subcharts
    x1 = np.linspace(0.0, 5.0)
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    x2 = np.linspace(0.0, 2.0)
    y2 = np.cos(2 * np.pi * x2)

    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
    fig.suptitle('A tale of 2 subplots')
    ax1.plot(x1, y1, 'o-')
    ax1.set_ylabel('Damped oscillation')
    ax2.plot(x2, y2, '.-')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Undamped')
    matplotlib.pyplot.show()

    # Assuming data is a pandas data frame
    # Histogram plotting
    data.plot.hist(bins=12, alpha=0.5)
    return


# TODO:
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
    charts(data)
    distributions()
    tests()

