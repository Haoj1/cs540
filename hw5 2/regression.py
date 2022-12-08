import random
import numpy as np
import csv
import math
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
    dataset = []

    with open(filename) as file:
        read = csv.reader(file, delimiter=",")
        row_num = 0
        for row in read:
            current = []
            if row_num != 0:
                for x in range(1, len(row)):
                    current.append(row[x])
                dataset.append(current)
            row_num += 1

    return np.asarray(dataset).astype(float)


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
    num_data = len(dataset)
    mean = 0
    sd = 0

    for x in dataset:
        mean += x[col]

    mean = mean/len(dataset)

    for x in dataset:
        sd += (x[col] - mean) ** 2

    sd = math.sqrt((1/(len(dataset)-1))*sd)

    print(num_data)
    print(round(mean, 2))
    print(round(sd, 2))


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
    mse_sum = 0

    for ind in dataset:
        count = 0
        res = betas[0]
        for x in range (1, len(betas)):
            res += betas[x]*ind[cols[count]]
            count += 1

        mse_sum += (res - ind[0]) ** 2

    mse = (1/len(dataset))*mse_sum
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

    grads = []

    for y in range (0, len(betas)):
        gd_sum = 0
        gd_curr = 0
        for ind in dataset:
            count = 0
            res = betas[0]

            for x in range(1, len(betas)):
                res += betas[x] * ind[cols[count]]
                count += 1

            if y != 0:
                gd_curr = (res - ind[0]) * ind[cols[y-1]]
                gd_sum += gd_curr

            else:
                gd_curr += res - ind[0]

        if y != 0:
            gd_sum = (2 / len(dataset)) * gd_sum

        else:
            gd_sum = (2 / len(dataset)) * gd_curr

        grads.append(gd_sum)

    return np.asarray(grads)


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

    betas_updated = betas

    for x in range (1, T+1):
        print_array = []
        print_array.append(x)
        beta_count = 0
        betas_updated_temp = []

        for y in range (0, len(betas_updated)):
            betas_updated_temp.append(betas_updated[y] - eta*(gradient_descent(dataset, cols, betas_updated)[beta_count]))
            beta_count += 1

        betas_updated = betas_updated_temp
        print_array.append(regression(dataset, cols, betas_updated))

        for beta in betas_updated:
            print_array.append(beta)

        print ( "%d %.2f %.2f %.2f %.2f" % (print_array[0], print_array[1], print_array[2], print_array[3], print_array[4]), end="")
        print ("")


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
    x = []
    y = []

    for ind in dataset:
        count = 0
        row = []
        row2 = []
        row2.append(ind[0])
        y.append(row2)
        row.append(1)
        for val2 in range (1, len(ind)):
            if val2 == cols[count]:
                row.append(ind[val2])

            if count == (len(cols)-1):
                break

            count += 1
        x.append(row)

    x_matrix = np.matrix(x)
    x_matrix_T = np.matrix.transpose(x_matrix)
    y_matrix = np.matrix(y)
    betas = np.dot(np.dot(np.linalg.inv(np.dot(x_matrix_T, x_matrix)), x_matrix_T), y_matrix)

    betas_array = []

    for index in range (0, len(betas)):
        betas_array.append(betas.item(index))

    mse = regression(dataset, cols, betas_array)
    return (mse, *betas_array)


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
    betas = compute_betas(dataset, cols)
    result = betas[1]

    for x in range (2, len(betas)):
        result += betas[x]*features[x-2]

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
    linear_dataset = []
    quad_dataset = []
    for n in range (0, len(X)):
        y = betas[0] + betas[1]*X[n][0] + np.random.normal(0, sigma)
        append_val = [y, X[n][0]]
        linear_dataset.append(append_val)

    for n in range (0, len(X)):
        y = alphas[0] + alphas[1]*((X[n][0]) ** 2) + np.random.normal(0, sigma)
        append_val = [y, X[n][0]]
        quad_dataset.append(append_val)

    return np.asarray(linear_dataset), np.asarray(quad_dataset)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = []
    beta_couple = []
    alpha_couple = []
    sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 10**2, 10**3, 10**4, 10**5]
    synth_sets_tuples = []
    mse_list = []

    for x in range (0, 1000):
        input = [random.randint(-100,100)]
        X.append(input)

    for x in range (0, 2):
        alpha_couple.append(random.random() * 10)
        beta_couple.append(random.random() * 10)

    for sigma in sigmas:
        synth_sets_tuples.append(synthetic_datasets(beta_couple, alpha_couple, X, sigma))

    for x in range (0, len(synth_sets_tuples)):
        mse_list.append([compute_betas(synth_sets_tuples[x][0], cols=[1])[0], compute_betas(synth_sets_tuples[x][1], cols=[1])[0]])

    plt.plot(sigmas, mse_list, "-o")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("MSE")
    plt.xlabel("Sigmas")
    plt.legend(["Linear Model", "Quadratic Model"])
    plt.savefig( "mse.pdf", format="pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    dataset = get_dataset('bodyfat.csv')
    print(gradient_descent(dataset, cols=[2, 3], betas=[0, 0, 0]))
    plot_mse()

