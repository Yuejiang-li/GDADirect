import numpy as np


class Adjlist:
    def __init__(self, adj_mat) -> None:
        self.n = adj_mat.shape[0]
        self.neighbors = {}
        self.weights = {}

        for i in range(self.n):
            self.neighbors[i] = np.where(adj_mat[i] != 0)[0]
            self.weights[i] = adj_mat[i][self.neighbors[i]]

        self.degree = np.sum(adj_mat, axis=1)
        self.rads = np.sum(np.abs(adj_mat), axis=1)
        self.leftend = np.min(self.degree - self.rads)
