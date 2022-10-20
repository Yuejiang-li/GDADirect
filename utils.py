import numpy as np
import networkx as nx


def add_one_nodes(W_):
    n = W_.shape[0]
    n = n + 1
    W = np.zeros((n, n))
    W[1:, 1:] = W_
    W[1:, 0] = 1.0
    src = 1 + np.random.randint(n-1)
    W[0, src] = 1.0

    return W


def gen_valid_graph(graph_type, n, **args):
    if graph_type == 'random':
        p = args['p']
        G = nx.gnp_random_graph(n-1, p, directed=True)
    elif graph_type == 'scale-free':
        G = nx.scale_free_graph(n-1)
    elif graph_type == 'gn':
        G = nx.gn_graph(n-1)
    W_ = nx.to_numpy_array(G)
    W_[W_ > 0] = 1.0
    W = add_one_nodes(W_)
    return W
