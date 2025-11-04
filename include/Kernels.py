import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from scipy.special import xlogy
from numba import njit


def n_exponential(X, Y, ell, pointwise=True):
    if pointwise:
        r = np.sum(np.square(X - Y), axis=1)
    else:
        r = cdist(X, Y, 'sqeuclidean')
    return np.exp(-1 * (np.sqrt(np.squeeze(r)) / np.squeeze(ell)))

def n_squared_exponential(X, Y, ell, pointwise=True):
    if pointwise:
        r = np.sum(np.square(X - Y), axis=1)
    else:
        r = cdist(X, Y, 'sqeuclidean')
    return np.exp(-1 * (np.squeeze(r) / np.squeeze(ell) ** 2))

def n_matern(X, Y, ell=1, nu=1, pointwise=True):
    eps = 1E-14
    ell = np.squeeze(ell)
    l = ell
    v = nu
    if pointwise:
        r = np.sum(np.square(X - Y), axis=1)
    else:
        r = cdist(X, Y, 'sqeuclidean')
    r = np.sqrt(r)
    return np.where(r <= eps, 1.0, (2 ** (1 - v) / gamma(v))*
                    ((np.sqrt(2 * v) * r / l) ** v)*
                    (kv(v, np.sqrt(2 * v) * r / l)) )

def n_multiquadrics(X, Y, ell, pointwise=True):
    if pointwise:
        r = np.sum(np.square(X - Y), axis=1)
    else:
        r = cdist(X, Y, 'sqeuclidean')
    return np.sqrt((r / (np.squeeze(ell) ** 2)) + 1)

def n_thin_plate_spline(X, Y, ell, pointwise=True):
    if pointwise:
        r = np.sum(np.square(X - Y), axis=1)
    else:
        r = cdist(X, Y, 'sqeuclidean')
    ell = np.power(np.squeeze(ell), 2)
    return xlogy(r/ell, r/ell)


def vec_kernel(M, _ker, src_c_mat, trg_c_mat, param_c_mat):
    param_dim = param_c_mat.shape[0]
    src, trg, par = my_map(src_c_mat, trg_c_mat, param_c_mat, M)
    if param_dim == 1:
        return _ker(src, trg, par)
    elif param_dim == 2:
        return _ker(src, trg, par[:, 0], par[:, 1])

@njit(cache=True, fastmath=True)
def my_map(src_c_mat, trg_c_mat, param_c_mat, M):
    evals = M.shape[0]
    d = src_c_mat.shape[0]
    param_dim = param_c_mat.shape[0]
    src = np.zeros((evals, d))
    trg = np.zeros((evals, d))
    par = np.zeros((evals, param_dim))
    for i in range(d):
        src[:, i] = src_c_mat[i, M[:, i]]
    for i in range(d):
        trg[:, i] = trg_c_mat[i, M[:, i + d + param_dim]]
    for i in range(param_dim):
        par[:, i] = param_c_mat[i, M[:, i + d]]
    return src, trg, par

def nf_map(M, src, trg, param_c_mat, kernel):
    idx = M[:, 0]
    shape = (src.shape[0], trg.shape[0])
    multi_idx = np.array(np.unravel_index(idx, shape, order='C')).T
    src = src[multi_idx[:, 0], :]
    trg = trg[multi_idx[:, 1], :]
    if param_c_mat.shape[0] == 1:
        A = param_c_mat[0, M[:, 1]]
        return kernel(src, trg, A)
    if param_c_mat.shape[0] == 2:
        A = param_c_mat[0, M[:, 1]]
        B = param_c_mat[1, M[:, 2]]
        return kernel(src, trg, A, B)


def form_kernel_matrix(kernel_name, source, target, params):
    match kernel_name:
        case "exponential":
            return n_exponential(source, target, params[0], False)
        case "matern":
            return n_matern(source, target, params[0], params[1], False)
        case "thin_plate_spline":
            return n_thin_plate_spline(source, target, params[0], False)
        case "multiquadric":
            return n_multiquadrics(source, target, params[0], False)
        case "squared_exponential":
            return n_squared_exponential(source, target, params[0], False)
