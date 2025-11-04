import numpy as np
from numba import jit, prange

@jit(nopython=True, cache=True)
def create_barycentric_weights(n):
    w = np.zeros(n)
    for j in range(n):
        w[j] = ((-1)**j * np.sin((2*j + 1)*np.pi/(2*n)))
    return w

@jit(nopython=True, cache=True)
def eval_lagrange_basis_barycentric(x, nodes, weights, k):
    num = weights[k]/(x - nodes[k])
    denom = 0.0
    for j in range(len(nodes)):
        denom += weights[j]/(x - nodes[j])
    return num/denom

@jit(nopython=True, cache=True)
def eta_k(n, k):
    return np.cos(np.pi * (2 * k - 1)/ (2 * n ))

@jit(nopython=True, cache=True)
def I(x, a, b):
    return (x + 1.0)*(b - a)/2.0 + a

@jit(nopython=True, cache=True)
def create_chebyshev_nodes(a, b, n):
    nodes = np.zeros(n)
    for i in range(n):
        nodes[i] = I(eta_k(n, i + 1), a, b)
    return nodes

@jit(nopython=True, fastmath=True, cache=True)
def create_node_map_mats(bounding_boxes, n):
    dim = bounding_boxes.shape[0]
    map_mat = np.zeros((dim, n))
    for i in range(dim):
        map_mat[i, :] = create_chebyshev_nodes(bounding_boxes[i, 0], bounding_boxes[i, 1], n)
    return map_mat

@jit(nopython=True, fastmath=True, cache=True)
def create_basis_matrices(points, bounding_boxes, n):
    num_points = points.shape[0]
    dim = points.shape[1]
    basis_mats = np.zeros((num_points, n, dim))
    weights = create_barycentric_weights(n)
    for i in range(dim):
        a = bounding_boxes[i, 0]
        b = bounding_boxes[i, 1]
        nodes = create_chebyshev_nodes(a, b, n)
        for j in prange(num_points):
            p = points[j, i]
            denom = 0
            for k in range(n):
                denom += weights[k] / (p - nodes[k])
            for k in range(n):
                num = weights[k]/(p - nodes[k])
                basis_mats[j, k, i] =  num/denom

    return basis_mats


