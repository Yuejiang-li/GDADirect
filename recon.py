"""
Reconstruction algorithms.
"""
import numpy as np
from scipy.sparse.linalg import cg


def smooth_recon(seeds, M, y, recon_method='cg'):
    """
    Reconstruct graph signal in a semi-supervised fashion, i.e.,
    x = arg min_x |Hx - y|_2^2 + \mu reg(x).
    Here, reg(x) = x^T L x for undirected graph, and x^T L^T L x.
    Thus, for parameter M matrix, mu * L should be passed for undirected graph,
    and mu * L^T L should be passed for directed graph.
    The best estimates of x is x^\hat = (A + M) ^ {-1} H^T y
    
    @params:
        seeds (iterative): indicate the index of the sampled nodes.
        M (np.array, n*n): matrix in regularizer. L for undirected graphs, and L^T @ L
            for directed ones.
        y (np.array, k): observed signals at sampled nodes.
        recon_method (str): the algorithm of reconstruction, 'cg' by default.
    """
    n = len(M)
    a = np.zeros(n)
    a[seeds] = 1.0
    x_hat = np.zeros(n)
    x_hat[seeds] = y    # this is H^T * y
    if recon_method == 'inv':
        x_hat = np.linalg.inv(np.diag(a) + M) @ x_hat
    elif recon_method == 'solve':
        x_hat = np.linalg.solve(np.diag(a) + M, x_hat)
    else:
        # conjugate gradient 'default'
        x_hat, exit_code = cg(np.diag(a) + M, x_hat)

    return x_hat
