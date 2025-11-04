import time
import gc
import sys
import numpy as np
from py_markdown_table.markdown_table import markdown_table

from include.ClusterTree import ClusterTree
from include.BlockClusterTree import BlockTree
from include.HMatrix import *
from include.Kernels import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Problem setup
dim = 3
adm_param = np.sqrt(dim)
kernels = ["matern"]
sizes = ["small", "medium", "large"]
epsilon = 1e-5
p_sz_lst = [4, 5, 6]

param_ell = np.load("Samples/param_ell_extended.npy")
param_nu = np.load("Samples/param_nu.npy")
num_samples = len(param_ell)

low_ell = .1
high_ell = 1.0
param_domains = {
    'exponential': np.array([[low_ell, high_ell]]),
    'thin_plate_spline': np.array([[low_ell, high_ell]]),
    'squared_exponential': np.array([[low_ell, high_ell]]),
    'multiquadric': np.array([[low_ell, high_ell]]),
    'matern': np.array([[low_ell, high_ell], [0.5, 3]]),
}

online_data = []

for kernel in kernels:
    counter = 0
    for size_label in sizes:
        random_matrix = np.load(f"Samples/{size_label}_matrix.npy")
        sz = random_matrix.shape[0]
        x0 = np.load(f"Samples/{size_label}_x0.npy")
        idx = np.load(f"Samples/{size_label}_idx.npy")

        p_sz = p_sz_lst[counter]
        counter += 1
        lvl_max = p_sz - 2

        eprint(f"\n[size: {size_label}] HMatrix run for kernel: {kernel}, n={sz}, lvl_max={lvl_max}")
        param_domain = param_domains[kernel]

        T_left = ClusterTree(random_matrix, lvl_max)
        T_right = ClusterTree(random_matrix, lvl_max)
        Block_T = BlockTree(T_left, T_right, random_matrix, adm_param)

        PH = HMatrix(kernel, Block_T, ep=epsilon)

        ff_times_ph, nf_times_ph, mvm_times_ph, err_ph_list = [], [], [], []

        for i in range(num_samples):
            if kernel == 'matern':
                param = np.array([param_ell[i], param_nu[i]])
            else:
                param = np.array([param_ell[i]])

            ff, nf = PH.online_mode(param)
            ff_times_ph.append(ff)
            nf_times_ph.append(nf)

            K_rows = form_kernel_matrix(kernel, random_matrix[idx], random_matrix, param)
            y_true = K_rows @ x0

            t0 = time.perf_counter()
            y_approx = PH.mvm(x0)[idx]
            mvm_times_ph.append(time.perf_counter() - t0)

            rel_err = np.linalg.norm(y_true - y_approx) / np.linalg.norm(y_true)
            err_ph_list.append(rel_err)


        r_mean = PH.get_mean_rank()

        online_data.append({
            "Kernel": kernel,
            "size": size_label,
            "nf time": float(np.mean(nf_times_ph)),
            "ff time": float(np.mean(ff_times_ph)),
            "rank": r_mean,
            "mvm time": float(np.mean(mvm_times_ph)),
            "error": float(np.mean(err_ph_list)),
            "ep": epsilon,
        })

        PH = None
        Block_T = None
        gc.collect()



eprint("\n===== ONLINE METRICS TABLE =====")
markdown = markdown_table(online_data).get_markdown()
eprint(markdown)