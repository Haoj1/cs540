import csv
import math
import random

import numpy as np
import numpy.random
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = None
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            dataset.append([float(line[features]) for features in line if features != 'IDNO'])
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    num = len(dataset)
    sum = 0.0
    print(num)
    for i in range(num):
        sum += dataset[i][col]
    mean = sum / num
    print("%.2f" % mean)
    deviation = math.sqrt(math.fsum(math.pow(dataset[i][col]-mean, 2) for i in range(num)) / (num - 1))
    print("%.2f" % deviation)
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = None
    y = [features[0] for features in dataset]
    f = []
    for i in range(len(dataset)):
        f.append(math.fsum(dataset[i][cols[j-1]]*betas[j] for j in range(len(betas))[1:])+betas[0])
    mse = (math.fsum(math.pow(f[i]-y[i], 2) for i in range(len(dataset)))) / len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = None
    grads = []
    y = [features[0] for features in dataset]
    f = []
    for i in range(len(dataset)):
        f.append(math.fsum(dataset[i][cols[j-1]]*betas[j] for j in range(len(betas))[1:])+betas[0])
    grads.append(math.fsum(f[i] - y[i] for i in range(len(dataset))) * 2 / len(dataset))
    for beta in range(len(cols)):
        grads.append((math.fsum((f[i]-y[i])*dataset[i][cols[beta]] for i in range(len(dataset)))) * 2 / len(dataset))
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    for i in range(T):

        grads = gradient_descent(dataset, cols, betas)
        betas = [(betas[j] - eta * grads[j]) for j in range(len(betas))]
        mse = regression(dataset, cols, betas)
        print(i+1, end=' ')
        print("%.2f" % mse, end='')
        for beta in betas:
            print(" %.2f" % beta, end='')
        print()
    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = None
    mse = None
    y = np.array([features[0] for features in dataset]).transpose()
    x = []
    x.append([1 for i in range(len(dataset))])
    for i in range(len(cols)):
        x.append(dataset[:, cols[i]])
    x = np.array(x)
    x = x.transpose()
    betas = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose()), y)
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    result = None
    betas = compute_betas(dataset, cols)
    result = math.fsum(betas[i+2]*features[i] for i in range(len(features))) + betas[1]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear = np.zeros([len(X), 2])
    quadratic = np.zeros([len(X), 2])
    X = np.array(X)
    betas = np.array(betas)
    alphas = np.array(alphas)
    for i in range(len(X)):
        linear[i][0] = betas[0] + math.fsum((betas[j] * X[i][0]) for j in range(len(betas))[1:]) \
                       + numpy.random.normal(0, scale=sigma)
    linear[:, 1] = X[:, 0]
    for i in range(len(X)):
        quadratic[i][0] = alphas[0] + math.fsum((alphas[j] * math.pow(X[i][0], 2)) for j in range(len(alphas))[1:]) \
                       + numpy.random.normal(0, scale=sigma)
    quadratic[:, 1] = X[:, 0]
    return linear, quadratic


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = [[random.randint(-100, 100)] for i in range(1000)]
    beta_couple = []
    alpha_couple = []
    linear_dataset = []
    quadratic_dataset = []
    linear_mse = []
    quadratic_mse = []
    sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    while len(beta_couple) < 2:
        beta = random.random() * 10
        if beta != 0:
            beta_couple.append(beta)
    while len(alpha_couple) < 2:
        alpha = random.random() * 10
        if alpha != 0:
            alpha_couple.append(alpha)
    for sigma in sigmas:
        linear, quadratic = synthetic_datasets(beta_couple, alpha_couple, X, sigma)
        linear_dataset.append(linear)
        quadratic_dataset.append(quadratic)
    for i in range(len(linear_dataset)):
        linear_mse.append(compute_betas(linear_dataset[i], cols=[1])[0])
        quadratic_mse.append(compute_betas(quadratic_dataset[i], cols=[1])[0])

    plt.plot(sigmas, linear_mse, "-o")
    plt.plot(sigmas, quadratic_mse, "-o")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("MSE of Trained Model")
    plt.xlabel("Standard Deviation of Error Term")
    plt.legend(["MSE of Linear Dataset", "MSE of Quadratic Dataset"])
    plt.savefig("mse.pdf", format="pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
