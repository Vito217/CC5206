import numpy as np


def pca(array, dimension):
    array = array.astype(np.float64) - np.mean(array, axis=0)
    covariance = np.cov(array, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    w = np.concatenate((eigenvectors, np.reshape(eigenvalues, (-1, 1))), axis=1)
    w = w[-w[:, -1].argsort()]
    w = np.flip(w, axis=0)
    w = w[:, :w.shape[1]-1]
    w = w[:dimension]
    arr = np.dot(array, np.transpose(w))
    return arr
