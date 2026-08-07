import random
import sys
import numpy as np



from include.ClusterTree import ClusterTree
from include.BlockClusterTree import BlockTree
from include.HMatrix import *
from include.H2Matrix import  *
from include.Kernels import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# problem setup
dim = 3 # dimension of the kernel
num_nodes = 15 # number of spatial Chebyshev nodes
num_param_nodes = 27 # number of parameter Chebyshev nodes
sz = 8**5 # the size of the kernel matrix
adm_param = np.sqrt(dim) # admissibility parameter
# kernel_names = ["exponential", "thin_plate_spline", "squared_exponential", "multiquadric", "matern"]
kernel_name = "exponential" # we will pick the exponential kernel
tolerance = 1e-5 # tolerance given to our method
param_space =  np.array([[.25, 1.0]]) # the values where \ell will vary in for the exponential kernel
domain = np.random.rand(sz, dim) # construct a  three-dimensional random ground set
lvl_max = 3 # value for l_{\max}

# construct the left and right cluster trees
T_left = ClusterTree(domain, lvl_max)
T_right = ClusterTree(domain, lvl_max)
# construct the block cluster tree
Block_T = BlockTree(T_left, T_right, domain, adm_param)

# Let's form a parametric H-matrix
# to form a Paramtric H^{2}-matrix use the following command:
# PH = ParamH2Matrix(kernel_name, Block_T, num_nodes, param_space, param_nodes=num_param_nodes, ep=tolerance)
PH = ParamHMatrix(kernel_name, Block_T, num_nodes, param_space,
                  param_nodes=num_param_nodes, ep=tolerance)
# perform the offline stage.
PH.offline_mode()

# sample 10 random parameters from the parameter space
param_ell = np.random.uniform(param_space[0][0], param_space[0][1], 10)

# a list to contain our errors
err_lst = []
# sample some random points to calculate a proxy  mvm error
idx = np.random.choice(sz, 200, replace=False)

for param in param_ell:
    # we have to encase the param in a vector
    param_vec = np.array([[param]], dtype=np.float64).reshape(1, 1)
    # perform the online stage for a particular parameter
    PH.online_mode(param_vec)

    # generate a random vector
    x0 = np.random.randn(sz)

    # obtain the ground truth mvm vector.
    K_rows = form_kernel_matrix(kernel_name, domain[idx], domain, np.array([param]))
    y_true = K_rows @ x0

    # obtain the approximate mvm vector using our method.
    y_approx = PH.mvm(x0)[idx]

    # compute the relative error
    rel_err = np.linalg.norm(y_true - y_approx) / np.linalg.norm(y_true)
    err_lst.append(rel_err)

# display the mean error.
print(np.mean(err_lst))




