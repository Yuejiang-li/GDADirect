import time
import scipy.sparse.linalg as sla
import numpy as np


def iter_greedy(M, k):
    """
    For optimization function that 
    max_{diag{a}} lambda_min(diag(a) + M)
    s.t.    a_i = 0, 1
            a_0 + ... + a_n = k
    iteratively select a_i = 0, so that the objective
    is maximized in the current round.
    """
    n = M.shape[0]
    a = np.zeros(n)
    seeds = []
    for i in range(k):
        max_val = -1.0
        max_idx = -1
        for j in range(n):
            if a[j] == 0:
                a[j] = 1.0
                B = np.diag(a) + M
                D, _ = sla.eigs(B, 1, which='SM')
                if np.abs(D[0]) > max_val:
                    max_val, max_idx = np.abs(D[0]), j
                a[j] = 0.0
        a[max_idx] = 1
        seeds.append(max_idx)
    
    return seeds

def bandlim_greedy(sig_base, k, k_list=None):
    """
    Implement greedy sampling strategy in "Discrete signal processing on graphs: Sampling Theory".
    @params:
        sig_base (np.array) : signal bases
        k (int) : sampling budget
    """
    start_time = time.time()
    n = sig_base.shape[0]
    seeds = []
    s_seeds = set([])
    if k_list is not None:
        run_time = np.zeros(len(k_list))
        k2idx = {k: idx for idx, k in enumerate(k_list)}
    else:
        run_time = None

    for cur_k in range(1, k+1):
        max_idx = -1
        max_val = -1.0
        for i in range(n):
            if i not in s_seeds:
                seeds.append(i)
                V = sig_base[seeds, :]
                S = np.linalg.svd(V, compute_uv=False)
                small_singval = np.min(S)
                if small_singval > max_val:
                    max_idx = i
                    max_val = small_singval
                seeds.pop()
        seeds.append(max_idx)
        s_seeds.add(max_idx)
        if k_list is not None and cur_k in k2idx:
            idx = k2idx[cur_k]
            run_time[idx] = time.time() - start_time
    
    return seeds, run_time


# if __name__ == "__main__":
#     sig_base = np.random.randn(20, 8)
#     k_list = [2, 4, 6, 8, 10]
#     seeds, run_time = bandlim_greedy(sig_base, max(k_list), k_list)
#     print(run_time)