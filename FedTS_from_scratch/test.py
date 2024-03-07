import numpy as np 
import cvxpy as cp
import torch

def optimizing_graph_matrix_neighbor(graph_matrix, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    # P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()

        graph_matrix[i, :] = torch.Tensor(x.value)
    return graph_matrix

