import argparse
import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from adj_list import Adjlist
from gershgorin_sample import DigraphGDASample, SignedSample
from spectral_proxy_sample import spectral_proxy_sample
from greedy_methods import bandlim_greedy
from sdp_sample import sdp_sample
from scipy.sparse.linalg import lobpcg
from utils import gen_valid_graph

colors = plt.cm.tab10(list(range(10)))
markers = ['o', 's', '*', '^', 'd', 'p']


def parse_args():
    parse = argparse.ArgumentParser(description='Running time experiments on random digraph.')
    parse.add_argument('--n_list', type=int, nargs='+', help="sizes of graphs to be test")
    parse.add_argument('--k_ratio', type=float, help='sampling buget K = k_ratio * n')
    parse.add_argument('--graph_rep', type=int, help='repeated times of graphs')
    parse.add_argument('--mu', type=float, help='reconstruction hyper-parameter')
    parse.add_argument('--eps', type=float, default=1e-5, help='error tolerence for GDA algorithm')
    parse.add_argument('--p', type=float, default=0.1, help='parameter for random graph generation.')
    parse.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
        help='whether to shuffle for GDA algorithm')

    args = parse.parse_args()
    return args


def repeat_exp_runtime(n_list, k_ratio, mu, graph_rep, p, eps=1e-5, **kwargs):
    run_time = np.zeros((6, len(n_list)))
    for i, n in enumerate(n_list):
        print(f"n = {n}")
        k = round(n * k_ratio)
        for _ in range(graph_rep):
            # Generate graphs
            W = gen_valid_graph('random', n, p=p)
            weights = np.random.rand(n, n)
            W[W > 0] = weights[W > 0]
            W = W / np.sum(W, axis=-1, keepdims=True)
            adjlist = Adjlist(W)

            start_time = time.time()
            L = np.eye(n) - W   # laplacian of the directed graph
            L_tilde = L.T @ L   # tilde_laplacian = L^T * L
            w, U = np.linalg.eig(L_tilde)   # eigen-basis of L_tilde
            od = np.argsort(w)
            U = ((U.T)[od]).T
            decon_time = time.time() - start_time
            # calculate second smallest eig
            X = np.random.randn(L_tilde.shape[0], 2)
            lbd_tilde, _ = lobpcg(L_tilde, X, largest=False)
            second_lbd = lbd_tilde[1]

            # Sample with different methods
            rand_seeds, gda_dir_seeds, sdp_seeds, gda_bal_seeds = {}, {}, {}, {}
            # random
            start_time = time.time()
            rand_seeds[k] = random.sample(range(n), k)
            end_time = time.time()
            run_time[0][i] += end_time - start_time

            # gda directed
            start_time = time.time()
            ep = second_lbd * n / k
            gda_dir_seeds[k] = list(DigraphGDASample(adjlist, k, ep, W, mu, eps=eps, **kwargs))
            end_time = time.time()
            run_time[2][i] += end_time - start_time
            print(f"GDA Direct time: {end_time - start_time}")

            # sdp relax with n <= 500 to avoid exceeding memory overhead.
            if n <= 500:
                start_time = time.time()
                sdp_seeds[k] = sdp_sample(mu * L_tilde, k)
                end_time = time.time()
                run_time[3][i] += end_time - start_time
                print(f"SDP Relax time: {end_time - start_time}")

            # gda balance
            start_time = time.time()
            gda_bal_seeds[k] = list(SignedSample(L_tilde, np.eye(n), k, mu=mu, eps=1e-3, shuffle=kwargs['shuffle']))
            end_time = time.time()
            run_time[4][i] += end_time - start_time
            print(f"GDA Balance time: {end_time - start_time}")

            # greedy
            start_time = time.time()
            _ = bandlim_greedy(U, k)
            end_time = time.time()
            run_time[1][i] += end_time - start_time + decon_time
            print(f"Greedy time: {end_time - start_time + decon_time}")
            
            # spectral proxies
            start_time = time.time()
            _ = spectral_proxy_sample(L_tilde, k)
            end_time = time.time()
            run_time[5][i] += end_time - start_time
            print(f"Spectral proxy time: {end_time - start_time}")
            
    run_time = run_time / graph_rep

    return run_time


if __name__ == "__main__":
    args = parse_args()
    args = args.__dict__
    print(args)
    jstr_args = json.dumps(args)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    run_time = repeat_exp_runtime(**args)
    np.savez_compressed(f"{current_time}.npz", run_time=run_time, config=jstr_args)
    run_time = np.where(run_time, run_time, np.nan)

    # visualize results
    fig = plt.figure(figsize=(5.625, 3.8), dpi=500)

    for i in range(1, run_time.shape[0]):
        plt.plot(args['n_list'], run_time[i], linewidth=1.5, color=colors[i], marker=markers[i],
            markerfacecolor='none', markersize=5)

    plt.yscale("log")
    plt.xlabel("Network Size: N")
    plt.ylabel("Running Time: seconds")
    plt.tight_layout()
    plt.grid(linestyle=':', linewidth=1.0)
    plt.savefig(f"{current_time}.png", dpi=500)
