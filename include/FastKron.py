import numpy as np
import numpy.random as npr
from functools import reduce
#########################################
###
### code from: https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
###
# Goal
# ----
# Compute (As[0] kron As[1] kron ... As[-1]) @ v

# ==== HELPER FUNCTIONS ==== #

    # Goal
    # ----
    # Compute (As[0] kron As[1] kron ... As[-1]) @ v

    # ==== HELPER FUNCTIONS ==== #

def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.

    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape

    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def refold(vec, mode, dims):
    """
    Refolds vector into tensor.

    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape

    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)


# ==== KRON-VEC PRODUCT COMPUTATIONS ==== #

def kron_vec_prod(As, v, transpose=False):
    """
    Computes (As[0] kron As[1] kron ... As[-1]) @ v efficiently,
    returning the result as a column vector of shape (k, 1).
    """
    if transpose:
        As = transpose_list(As)

    dims = [A.shape[0] for A in As]
    vt = v.reshape(dims)
    for i, A in enumerate(As):
        vt = refold(A @ unfold(vt, i, dims), i, dims)
    return vt.reshape(-1, 1)


def kron_brute_force(As, v):
    """
    Computes kron-matrix times vector by brute
    force (instantiates the full kron product).
    """
    return reduce(np.kron, As) @ v


def transpose_list(matrices):
    """
    Returns a new list where each matrix in `matrices` is transposed.

    Parameters:
        matrices (list of np.ndarray): List of 2D NumPy arrays.

    Returns:
        list of np.ndarray: Transposed matrices.
    """
    return [A.T for A in matrices]



