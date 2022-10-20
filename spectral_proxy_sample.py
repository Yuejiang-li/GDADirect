"""
Implement spectral proxy based sampling algorithm in 
"(TSP'16) Efficient Sampling Set Selection for Bandlimited
Graph Signals Using Graph Spectral Proxies"
"""
import time
import numpy as np


def spectral_proxy_sample(L, k, power=100, k_list=None):
    """
    @ params:
        L (N * N, numpy.array) : the variation operator.
        M (int) : sampling buget.
        k (int) : degree of the spectral proxy.
    """
    start_time = time.time()
    n = len(L)
    L_power = np.copy(L)
    for i in range(power-1):
        L_power = L_power @ L
    
    L_top = L_power.T @ L_power
    
    samples = []

    if k_list is not None:
        run_time = np.zeros(len(k_list))
        k2idx = {k: idx for idx, k in enumerate(k_list)}
    else:
        run_time = None

    for cur_k in range(1, k+1):
        S_c_idx = [x for x in range(n) if x not in samples]
        L_r = L_top[S_c_idx][:, S_c_idx]
        # ----- eigen decompostion -----
        w, V = np.linalg.eig(L_r)
        w, V = w.real, V.real
        min_idx = np.argmin(w)
        psi = V[:, min_idx]
        # ------------------------------
        # ----- LOBPCG version -----
        # During experiments, we found LOBPCG fails to generate
        # true eigenvectors, so we deprecate this method. 
        # X = np.random.randn(L_r.shape[0], 1)
        # _, psi = lobpcg(L_r, X, largest=False)
        # --------------------------
        psi = np.squeeze(psi)
        idx = np.argmax(psi * psi)
        samples.append(S_c_idx[idx])
        if k_list is not None and cur_k in k2idx:
            idx = k2idx[cur_k]
            run_time[idx] = time.time() - start_time

    return samples, run_time
