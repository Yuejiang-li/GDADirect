import argparse
import time
import random
import json
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import cvxopt
from datetime import datetime
from functools import partial
from adj_list import Adjlist
from gershgorin_sample import DigraphGDASample, SignedSample
from greedy_methods import bandlim_greedy
from spectral_proxy_sample import spectral_proxy_sample
from signals import gen_signal
from recon import smooth_recon
from sdp_sample import sdp_sample
from utils import gen_valid_graph
from scipy.sparse.linalg import lobpcg

colors = plt.cm.tab10(list(range(10)))
markers = ['o', 's', '*', '^', 'd', 'p']


def parse_args():
    parse = argparse.ArgumentParser(description='Reconstruction MSE experiments on digraph.')
    parse.add_argument('--size', type=int, help='size of network')
    parse.add_argument('--g_type', type=str, choices=['random', 'gn'], help='type of graph')
    parse.add_argument('--s_type', type=str, choices=['lowpass', 'normal', 'diffusion'], help='type of signal')
    parse.add_argument('--k_start', type=int, help='start of sampling budget K')
    parse.add_argument('--k_end', type=int, help='end of sampling budget K')
    parse.add_argument('--k_step', type=int, help='step of sampling budget K')
    parse.add_argument('--graph_rep', type=int, help='repeated times of graphs')
    parse.add_argument('--sig_rep', type=int, help='repeated times of signals')
    parse.add_argument('--sigma2', type=float, help='noise energy, set to 0.0 for noise-free')
    parse.add_argument('--mu', type=float, help='reconstruction hyper-parameter')
    parse.add_argument('--eps', type=float, default=1e-5, help='error tolerence for GDA algorithm')
    parse.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
        help='whether to shuffle for GDA algorithm')
    parse.add_argument('--para', type=int, default=0, help='parallelism, set to 0 for single core processing')
    parse.add_argument('--epr', type=float, default=None, help='epsilon * mu for GDA-direct algorithm')

    parse.add_argument('--topk_ratio', type=float, default=0.1,
        help='parameter for low-pass signal, the bandlimit is topk_ratio * N')
    parse.add_argument('--p', type=float, default=0.1, help='parameter for random graph generation.')

    parse.add_argument('--omega', type=float, default=0.001, help='parameter for GMRF signal')

    parse.add_argument('--T', type=int, default=30, help='diffusion steps for diffusion signal')
    parse.add_argument('--alpha', type=float, default=0.1, help='diffusion strength for diffusion signal')

    args = parse.parse_args()
    return args


def sanity_check(args):
    if args['g_type'] == 'random':
        if 'p' not in args:
            raise KeyError("arg p should be set if using g_type == random!")
    else:
        args.pop('p')
    
    if args['s_type'] == 'lowpass':
        if 'topk_ratio' not in args:
            raise KeyError("arg topk_ratio should be set if using s_type == lowpass!")
    else:
        args.pop('topk_ratio')

    if args['s_type'] == 'normal':
        if 'omega' not in args:
            raise KeyError("arg omega should be set if using s_type == normal!")
    else:
        args.pop('omega')
    
    if args['s_type'] == 'diffusion':
        if 'T' not in args:
            raise KeyError("arg T should be set if using s_type == diffusion!")
        if 'alpha' not in args:
            raise KeyError("arg alpha should be set if using s_type == diffusion!")
    else:
        args.pop('T')
        args.pop('alpha')


def recon_sig(k_list, rand_seeds, greedy_seeds, gda_dir_seeds, sdp_seeds, gda_bal_seeds,
    sp_seeds, mu, L_tilde, sig):
    signal, base_signal = sig[0], sig[1]
    res = np.zeros((6, len(k_list)))
    # recon with random seeds.
    for i, k in enumerate(k_list):
        cur_seeds = rand_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[0][i] = (d @ d)

    # recon with greedy seeds.
    for i, k in enumerate(k_list):
        cur_seeds = greedy_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[1][i] = (d @ d)

    # recon with GDAS-direct
    for i, k in enumerate(k_list):
        cur_seeds = gda_dir_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[2][i] = (d @ d)

    # recon with sdp-relaxation
    for i, k in enumerate(k_list):
        cur_seeds = sdp_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[3][i] = (d @ d)
#     print("sdp done.")
    
    # recon with GDAS-signed
    for i, k in enumerate(k_list):
        cur_seeds = gda_bal_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[4][i] = (d @ d)
    
    # recon with spectral proxy
    for i, k in enumerate(k_list):
        cur_seeds = sp_seeds[k]
        y = signal[cur_seeds]
        x_hat = smooth_recon(cur_seeds, mu * L_tilde, y)
        d = x_hat - base_signal
        res[5][i] = (d @ d)

    return res


def repeat_exp(size, k_list, mu, g_type, s_type,
    graph_rep, sig_rep, eps=1e-6, **kwargs):
    n = size
    total_start = time.time()
    res = np.zeros((6, len(k_list)))
    for graph_rep in range(graph_rep):
        # Generate graphs
        W = gen_valid_graph(g_type, n, **kwargs)
        weights = np.random.rand(n, n)
        W[W > 0] = weights[W > 0]
        W = W / np.sum(W, axis=-1, keepdims=True)
        adjlist = Adjlist(W)

        start_time = time.time()
        L = np.eye(n) - W   # laplacian of the directed graph
        L_tilde = L.T @ L   # tilde_laplacian = L^T * L
        w, U = np.linalg.eig(L_tilde)   # eigen-basis of L_tilde
        w, U = w.real, U.real
        od = np.argsort(w)
        U = ((U.T)[od]).T
        decon_time = time.time() - start_time
        if s_type == 'normal':
            # calculate cov matrix
            cov_mat = np.linalg.inv(L_tilde + kwargs['omega'] * np.eye(n))
        else:
            cov_mat = None
        # calculate second smallest eig
        X = np.random.randn(L_tilde.shape[0], 2)
        lbd_tilde, _ = lobpcg(L_tilde, X, largest=False)
        second_lbd = lbd_tilde[1]

        # Sample with different methods
        sample_start = time.time()
        gda_dir_seeds, sdp_seeds, gda_bal_seeds = {}, {}, {}
        for i, k in enumerate(k_list):
            # sdp relax
            sdp_seeds[k] = sdp_sample(mu * L_tilde, k)

            # gda directed
            if 'epr' not in kwargs or kwargs['epr'] is None:
                ep = second_lbd * n / k
            else:
                ep = kwargs['epr'] / mu
            # print(ep)
            gda_dir_seeds[k] = list(DigraphGDASample(adjlist, k, ep, W, mu, eps=eps, **kwargs))

        # random
        # seeds = random.sample(range(n), max(k_list))
        seeds = np.random.permutation(n)[:max(k_list)]
        rand_seeds = {k: seeds[:k] for k in k_list}

        # greedy
        greedy_all_nodes, _ = bandlim_greedy(U, max(k_list), k_list)
        greedy_seeds = {}
        for k in k_list:
            greedy_seeds[k] = greedy_all_nodes[:k]

        # spectral proxy
        sp_all_nodes, _ = spectral_proxy_sample(L_tilde, max(k_list), k_list=k_list)
        sp_seeds = {}
        for k in k_list:
            sp_seeds[k] = sp_all_nodes[:k]

        # gda balance
        gda_bal_all_seeds = list(SignedSample(L_tilde, np.eye(n), max(k_list), mu=mu, eps=eps, shuffle=kwargs['shuffle']))
        gda_bal_seeds = {k: gda_bal_all_seeds[:k] for k in k_list}

        print(f"Graph rep = {graph_rep}, sampling done, time consumes {time.time() - sample_start}")

        # signal reconstruction
        recon_start = time.time()
        if kwargs['para'] <= 1:
            for rep in range(sig_rep):
                base_signal, signal = gen_signal(n, s_type, U=U, W=W, cov_mat=cov_mat, **kwargs)
                res += recon_sig(k_list, rand_seeds, greedy_seeds, gda_dir_seeds, sdp_seeds,
                    gda_bal_seeds, sp_seeds, mu, L_tilde, (signal, base_signal))
        else:
            pool = mp.Pool(kwargs['para'])
            chunk_num = round(sig_rep / 20) + 1
            par_func = partial(recon_sig, k_list, rand_seeds, greedy_seeds, gda_dir_seeds, sdp_seeds,
                gda_bal_seeds, sp_seeds, mu, L_tilde)
            for c in range(chunk_num):
                sigs = []
                for i in range(20):
                    base_signal, signal = gen_signal(n, s_type, U=U, W=W, cov_mat=cov_mat, **kwargs)
                    sigs.append((signal, base_signal))
                ret = pool.map(par_func, sigs)
                for cr in ret:
                    res += cr
            pool.close()
            pool.join()
            
        print(f"Graph rep = {graph_rep}, recon. done, time consumes {time.time() - recon_start}")
        
    res = res / (graph_rep * sig_rep)

    print(f"Total running time: {(time.time() - total_start) / 60} min.")
    return res


if __name__ == "__main__":
    args = parse_args()
    args = args.__dict__
    # sanity check of args
    sanity_check(args)
    print(args)
    jstr_args = json.dumps(args)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # build k_list from args
    k_list = list(range(args['k_start'], args['k_end'], args['k_step']))

    mse_res = repeat_exp(k_list=k_list, **args)
    np.savez_compressed(f"{current_time}.npz", config=jstr_args, mse_res=mse_res)

    # visualize results
    fig = plt.figure(figsize=(5.625, 3.8), dpi=500)

    for i in range(mse_res.shape[0]):
        plt.plot(k_list, mse_res[i], linewidth=1.5, color=colors[i], marker=markers[i],
            markerfacecolor='none', markersize=5)

    plt.xlabel("Sampling Budget: K")
    plt.ylabel("Recon. MSE")
    plt.legend(["Random", "E-Optimal", "GDA-Direct", "SDP-Relax", "GDA-Balance", "SP"])
    plt.tight_layout()
    plt.grid(linestyle=':', linewidth=1.0)
    plt.savefig(f"{current_time}.png", dpi=500)
