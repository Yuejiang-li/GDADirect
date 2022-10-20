"""
Python implementation of GDAS.
"""

from collections import deque
import numpy as np
from scipy.sparse.linalg import lobpcg
from adj_list import Adjlist
from graph_balance_mod import constructGB


def cover_one_node(adjlist, mu, thres, p_hops, i):
    scale = np.ones(adjlist.n)
    visited = np.array([False for _ in range(adjlist.n)], dtype=np.bool)
    in_que = np.array([False for _ in range(adjlist.n)], dtype=np.bool)
    que = deque(maxlen=adjlist.n + 10)
    hops = np.zeros(adjlist.n, dtype=np.int)
    q_num = 0   # how much nodes are covered

    que.append(i)
    is_sampled = 1
    in_que[i] = True

    while que:
        idx = que.popleft()
        scale_tmp = np.where(scale >= 1.0, scale, 1.0)
        # calculate scale factor s_i, so that the left ends of disk-i is aligned as thre
        s_i = adjlist.degree[idx] * mu - thres + is_sampled
        is_sampled = 0.0
        neighbors = adjlist.neighbors[idx]
        r_i = mu * np.sum(np.divide(adjlist.weights[idx], scale_tmp[neighbors]))
        s_i = s_i / r_i
        scale[idx] = s_i

        if s_i >= 1.0 and hops[idx] <= p_hops:
            q_num += 1
            visited[idx] = True

            valid = neighbors[~in_que[neighbors]]
            # print(valid)

            que.extend(valid)
            in_que[valid] = True
            hops[valid] = hops[idx] + 1
        
        # print(f"que = {que}")
        # print(f"hops = {hops}")
        # print(f"scale = {scale}")
        # input()

    # set_size[i] = q_num
    return np.where(visited)[0]


def computeSets(adjlist: Adjlist, p_hops, mu, thres):
    """
    Given the threshold, for each node, calculate the coverage set.
    """
    # set_size = [0 for _ in range(adjlist.n)]
    cover_set = {}
    for i in range(adjlist.n):
        cover_set[i] = cover_one_node(adjlist, mu, thres, p_hops, i)

    return cover_set


def solSetCover(cover_set, adjlist, k, pebbles_order):
    """
    Greedily solve set cover problem.
    """
    selected_nodes = set([])
    selected_nodes_idx = []
    covered_nodes = set([])
    used_budget = 0

    while used_budget < k:
        while used_budget < k and len(covered_nodes) < adjlist.n:
            used_budget += 1
            left_num = {i: len(set(cover_set[i]) - covered_nodes) for i in pebbles_order if i not in selected_nodes}
            max_idx, _ = max(left_num.items(), key=lambda x: x[1])
            selected_nodes.add(max_idx)
            selected_nodes_idx.append(max_idx)
            covered_nodes |= set(cover_set[max_idx])
        
        if used_budget < k:
            cover_nums = np.zeros(adjlist.n)
            for sel_node in selected_nodes:
                cover_nums[list(cover_set[sel_node])] += 1
            min_cov_num = np.min(cover_nums)
            covered_nodes = set(np.where(cover_nums > min_cov_num)[0])

    covered_nodes = set([])
    for sel_node in selected_nodes:
        # print(cover_set[sel_node])
        # print(cover_set)
        covered_nodes |= set(cover_set[sel_node])

    return selected_nodes_idx, covered_nodes


def GDASample(adjlist: Adjlist, k, mu=0.01, eps=1e-5, shuffle=True):
    # left, right = 0, 1
    left, right = mu * adjlist.leftend, 1 + mu * adjlist.leftend
    thres = 0.5 * (right + left)
    p_hops = 12
    ans_seeds = None

    pebbles_order = list(range(adjlist.n))
    if shuffle:
        np.random.shuffle(pebbles_order)

    while abs(right - left) > eps:
        cover_set = computeSets(adjlist, p_hops, mu, thres)
        seeds, covers = solSetCover(cover_set, adjlist, k, pebbles_order)
        # cover_nums = [len(x) for i, x in cover_set.items()]
        # print(max(cover_nums))
        # print(thres, seeds, covers, f"left={len(covers) - adjlist.n}")
        # input()

        if len(covers) < adjlist.n:
            right = thres
        else:
            left = thres
            ans_seeds = seeds

        thres = (left + right) / 2

    if ans_seeds is None:
        # The given eps is too large, output a sub-optimal value.
        cover_set = computeSets(adjlist, p_hops, mu, thres)
        ans_seeds, covers = solSetCover(cover_set, adjlist, k, pebbles_order)

    return ans_seeds


def similarityTransform(LB_g):
    X = np.random.randn(LB_g.shape[0], 1)
    _, v = lobpcg(LB_g, X, largest=False)
    Lp = (LB_g / (v + 1e-6)) * np.squeeze(v)
    return Lp


def SignedSample(L_g, Sigma, k, mu=0.01, eps=1e-5, shuffle=True):
    """
    GDAS algorithm for signed graph. Reproduce paper (TPAMI'21) "Point Cloud
    Sampling via Graph Balancing and Gershgorin Disc Alignment".
    
    @params:
        L_g: the laplacian of the original graph.
        Sigma: the Sigma matrix to calculate trace(L_B * Sigma)
    """
    G = np.copy(L_g)
    # Assume self-loops exist. Also applies when self-loops don't exist
    for i in range(L_g.shape[0]):
        G[i][i] = - np.sum(G[i])
    G = - G
    # GB = constructGB(G, Sigma)
    GB = constructGB(G)
    LB_g = np.diag(np.sum(GB, axis=1)) - GB + np.diag(np.diag(GB))
    LP_g = similarityTransform(LB_g)
    GP = -(LP_g - np.diag(np.diag(LP_g)))
    adjlist = Adjlist(GP)
    return GDASample(adjlist, k, mu, eps, shuffle)


def DigraphGDASample(adjlist: Adjlist, k, ep, W, mu, eps=1e-5, **kwargs):
    # Calculate delta
    if 1 - ep * mu < 0:
        ep = 0.99 / mu
    delta = np.sqrt(1 - ep * mu)

    # Calculate c
    if k == 1:
        numerator = 3.0
    else:
        WT = W.T
        topk = np.sum(np.partition(WT, -(k-1))[:, -(k-1):], axis=-1)
        numerator = 3.0 + np.max(topk)
    n = W.shape[0]
    denominator = 1.0 * k * ep / n
    c = numerator / denominator

    # Calculate rho
    rho = 0.5 * (np.sqrt(np.square(delta * c) + 4 * mu) - delta * c)

    # perform GDAS
    mu_prime = rho / delta
    return GDASample(adjlist, k, mu=mu_prime, eps=eps, shuffle=kwargs['shuffle'])

