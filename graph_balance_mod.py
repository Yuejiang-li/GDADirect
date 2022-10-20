import numpy as np


def removeInconsitentEdges(W, color2idx, color_j, j):
    A = np.copy(W)
    # break positive edge:
    diff_color = np.zeros(len(W), dtype=np.bool)
    diff_color[color2idx[-color_j]] = True
    to_adjust_nodes = np.where((A[j] > 0) & diff_color)[0]
    A[j, to_adjust_nodes] = 0.0  # break positive edge between different colored nodes
    A[to_adjust_nodes, j] = 0.0

    # adjust negative edge:
    same_color = np.zeros(len(W), dtype=np.bool)
    same_color[color2idx[color_j]] = True
    to_adjust_nodes = np.where((A[j] < 0) & same_color)[0]
    if len(to_adjust_nodes) > 0:
        A[j, to_adjust_nodes] = 0.0
        A[to_adjust_nodes, j] = 0.0
        k = np.random.choice(color2idx[-color_j])
        tot_weights = 2 * np.sum(A[j][to_adjust_nodes])
        A[j][k] += tot_weights
        A[k][j] += tot_weights
        A[k, to_adjust_nodes] += 2 * A[j, to_adjust_nodes]
        A[to_adjust_nodes, k] += 2 * A[j, to_adjust_nodes]

    return A


def computefjbetaj(W, Sigma):
    LB = np.diag(np.sum(W, axis=1)) - W
    LBSigma = LB @ Sigma
    return np.trace(LBSigma)


def constructGB(adj_mat):
    """
    Given an unbalanced graph G, construct a corresponding balanced graph GB.
    """
    n = len(adj_mat)
    colors = {}
    color2idx = {1:[], -1:[]}
    setS = set()
    setC = set()

    # initialize
    idx = np.random.randint(0, adj_mat.shape[0] - 1)
    colors[idx] = 1
    color2idx[1].append(idx)
    # records the largest edge weights between set S and set C
    conn_CS_node = np.arange(n)
    conn_CS_weights = np.zeros(n)
    setS.add(idx)
    conn_CS_weights[idx] = -1.0
    new_2_C = np.where((adj_mat[idx] != 0) & (conn_CS_weights >= 0))[0]
    setC.update(new_2_C)
    adj_nodes = new_2_C[np.abs(adj_mat[idx][new_2_C]) > conn_CS_weights[new_2_C]]
    # print(f"adj_nodes = {adj_nodes}")
    conn_CS_weights[adj_nodes] = np.abs(adj_mat[idx][adj_nodes])
    conn_CS_node[adj_nodes] = idx

    #iteration
    while len(setC) > 0:
        # Pick the node with largest weights to set S
        C_nodes = list(setC)
        idx = C_nodes[np.argmax(conn_CS_weights[C_nodes])]
        conn_idx = conn_CS_node[idx]    # find the connected nodes in set S
        # decide color
        if adj_mat[idx][conn_idx] < 0:
            color_idx = -colors[conn_idx]
        else:
            same_color = np.zeros(n, dtype=np.bool)
            same_color[color2idx[colors[conn_idx]]] = True
            to_adjust_nodes = np.where((adj_mat[idx] < 0) & same_color)[0]
            if len(to_adjust_nodes) > 0 and len(color2idx[-colors[conn_idx]]) == 0:
                # There is possitive edge to remove, BUT there is no different colored nodes in set S.
                # In this case, we can directly set the newly added node to the different color.
                color_idx = -colors[conn_idx]
            else:
                color_idx = colors[conn_idx]
            
        adj_mat = removeInconsitentEdges(adj_mat, color2idx, color_idx, idx)
        colors[idx] = color_idx
        color2idx[color_idx].append(idx)

        setS.add(idx)
        setC.remove(idx)
        conn_CS_weights[idx] = -1.0
        new_2_C = np.where((adj_mat[idx] != 0) & (conn_CS_weights >= 0))[0]
        setC.update(new_2_C)
        adj_nodes = new_2_C[np.abs(adj_mat[idx][new_2_C]) > conn_CS_weights[new_2_C]]
        conn_CS_weights[adj_nodes] = np.abs(adj_mat[idx][adj_nodes])
        conn_CS_node[adj_nodes] = idx

    return adj_mat


if __name__ == "__main__":
    n = 20
    import networkx as nx
    G = nx.fast_gnp_random_graph(n, 0.2)
    W = nx.to_numpy_array(G)
    weights = np.random.randn(n, n)
    W[W > 0] = weights[W > 0]
    W = W + W.T
    W_b = constructGB(W)
    print(W_b)
    LB_g = np.diag(np.sum(W_b, axis=1)) - W_b + np.diag(np.diag(W_b))
    from gershgorin_sample import similarityTransform
    LP_g = similarityTransform(LB_g)
    center = np.diag(LP_g)
    rad = np.sum(np.abs(LP_g - np.diag(center)), axis=-1)
    left_ends = center - rad
    print(left_ends)
