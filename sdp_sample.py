"""
For problem
    max_{x} lambda_min( diag(x) + M )
    s.t.    1^T x = 1
            0 <= x <= 1,

we can transform it into the following sdp
    min_{x, t} -t
    s.t.    sum_i -E_i * x_i + I * t <= M
            1^T x = 1
            0 <= x <= 1,
and use cvxopt solver to solve it.
Note the constraint 0 <= x <= 1 can be simplified as x >= 0, 
which can be further written in LMI form
    sum_i -E_i * x_i <= 0 
"""

import numpy as np
from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
# for efficiency consideration (enough for good performance.)
solvers.options['abs_top'] = 1e-6


def sdp_sample(M, k):
    n = M.shape[0]
    c = matrix([0.0 for _ in range(n)] + [-1.0])

    # first constr. -E_i * x_i + I * t <= M
    val = [-1.0 for _ in range(n)] + [1.0 for _ in range(n)]
    E_col = []
    final_col = []
    for i in range(n):
        E_col.append(n*i + i)
        final_col.append(n*i + i)
    col = E_col + final_col
    row = list(range(n)) + [n for _ in range(n)]
    G = [spmatrix(val, row, col, (n+1, n*n)).T]
    h = [matrix(M)]

    # second constr. -E_i * x_i + 0 * t <= 0 
    row = list(range(n))
    val = [-1.0 for _ in range(n)]
    G.append(spmatrix(val, row, E_col, (n+1, n*n)).T)
    h.append(matrix(np.zeros((n, n))))

    # equility constr. 1^T x = 1
    A = matrix([[1.0] for _ in range(n)] + [[0.0]])
    b = matrix([1.0 * k])

    sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
    c_opt = np.array(sol['x']).squeeze()[:-1]   # relaxed continuous optimal solution
    od = np.argsort(c_opt)

    return od[-k:].tolist()


# if __name__ == "__main__":
#     M = np.random.randn(3, 3)
#     M = M.T @ M
#     seeds = sdp_sample(M, 2)
