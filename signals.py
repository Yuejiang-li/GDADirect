"""
Generate signals on graph.
"""
import numpy as np


def gen_signal(n, signal_type, **kwargs):
    if signal_type == 'lowpass':
        topk = round(kwargs['topk_ratio'] * n)
        base_signal = gen_smooth_sig(topk, 1.0, U=kwargs['U'])
    elif signal_type == 'normal':
        base_signal = gen_normal_signal(kwargs['cov_mat'])
    elif signal_type == 'diffusion':
        x0 = np.random.randn(n)
        if 'alpha' in kwargs:
            base_signal = gen_diffu_signal(kwargs['W'], x0, kwargs['T'], kwargs['alpha'])
        else:
            base_signal = gen_diffu_signal(kwargs['W'], x0, kwargs['T'])
    
    base_signal = base_signal - np.mean(base_signal)
    base_signal = base_signal / np.sqrt(base_signal @ base_signal)
    noise = np.sqrt(kwargs['sigma2'] / n) * np.random.randn(n)
    signal = base_signal + noise

    return base_signal, signal


def gen_normal_signal(cov_mat):
    n = cov_mat.shape[0]
    mean = np.zeros(n)
    return np.random.multivariate_normal(mean, cov_mat)


def gen_diffu_signal(W, x0, T, alpha=1.0):
    x = np.copy(x0)
    for i in range(T):
        x = (1-alpha) * x + alpha * W @ x
    
    return x


def gen_smooth_sig(topk, mag, M=None, U=None):
    """
    Generate bandlimited signal.
    @params:
        topk (int) : bandwidth of signal.
        mag (float) : the magnitude of the genrated signal.
        M (np.array, n*n) : matrix to define smooth regularizer.
            M = L if undirected graph, else L^T @ L for directed graph.
        U (np.array, n*n) : eigenvector array of M.
        Note: M and U cannot = None at the same time.
    """
    if M is None and U is None:
        raise ValueError("M and U are both None!")
    if U is None:
        w, U = np.linalg.eig(M)
        od = np.argsort(w)
        w = w[od]
        U = ((U.T)[od]).T
    base = U[:, :topk]
    weights = np.random.rand(topk)
    weights = weights / np.sum(weights)
    weights = np.sqrt(mag * weights)
    base_signal = base @ weights

    return base_signal


if __name__ == "__main__":
    W = np.random.rand(5, 5)
    W = W @ W.T
    x = gen_normal_signal(W)
    print(x)
    print(x.shape)