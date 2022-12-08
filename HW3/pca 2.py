from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    dataset = np.load(filename)
    dataset = np.array(dataset)
    average = np.mean(dataset, axis=0)
    dataset = dataset - average
    return dataset


def get_covariance(dataset):
    # TODO: add your code here
    transpose_set = np.transpose(dataset)
    covariance = np.dot(transpose_set, dataset) / (len(dataset) - 1)
    return covariance


def get_eig(S, m):
    # TODO: add your code here
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    w = np.flip(w)
    i = np.argsort(w)
    return np.diag(w), v[:, i]


def get_eig_perc(S, perc):
    # TODO: add your code here
    w, v = eigh(S)
    weight = np.sum(w) * perc
    w, v = eigh(S, subset_by_value=[weight, np.inf])
    w = np.flip(w)
    i = np.argsort(w)
    return np.diag(w), v[:, i]


def project_image(img, U):
    # TODO: add your code here
    alp = np.dot(np.transpose(U), img)
    x = np.dot(U, alp)
    return x


def display_image(orig, proj):
    # TODO: add your code here
    orig = np.reshape(orig, (32, 32))
    proj = np.reshape(proj, (32, 32))
    orig = np.transpose(orig)
    proj = np.transpose(proj)
    fig, (pic1, pic2) = plt.subplots(nrows=1, ncols=2)
    pic1.set_title("Original")
    pic2.set_title("Projection")

    plot1 = pic1.imshow(orig, aspect='equal')
    plot2 = pic2.imshow(proj, aspect='equal')
    fig.colorbar(plot1, ax=pic1)
    fig.colorbar(plot2, ax=pic2)
    plt.show()

if __name__ == "__main__":
    x = load_and_center_dataset('YaleB_32x32.npy')
    S = get_covariance(x)
    Lambda, U = get_eig(S, 2)
    projection = project_image(x[0], U)
    display_image(x[0], projection)

