import time
import gc
import sys
import numpy as np
from py_markdown_table.markdown_table import markdown_table

from include.ClusterTree import ClusterTree
from include.BlockClusterTree import BlockTree
from include.H2Matrix import *
from include.Kernels import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Problem setup
dim = 3
num_nodes = 8
num_param_nodes = 27
adm_param = np.sqrt(dim)
epsilon = 1e-5
kernels = ["multiquadric", "matern"]

sizes = ["small", "medium", "large"]
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
offline_data = []
online_data = []

for kernel in kernels:
    counter = 0
    for label in sizes:
        random_matrix = np.load(f"Samples/{label}_matrix.npy")
        sz = random_matrix.shape[0]
        x0 = np.load(f"Samples/{label}_x0.npy")
        idx = np.load(f"Samples/{label}_idx.npy")
        lvl_max = p_sz_lst[counter] - 2
        counter += 1
        eprint(f"\n[Size: {sz}] ParamHMatrix run for kernel: {kernel}")
        eprint(f"lvl_max = {lvl_max}")
        param_domain = param_domains[kernel]

        T_left = ClusterTree(random_matrix, lvl_max)
        T_right = ClusterTree(random_matrix, lvl_max)
        Block_T = BlockTree(T_left, T_right, random_matrix, adm_param)

        PH = ParamH2Matrix(kernel, Block_T, num_nodes, param_domain,
                          num_param_nodes=num_param_nodes, ep=epsilon, translation_invariant=True)

        offline_start = time.perf_counter()
        t1, t2 = PH.offline_mode()
        offline_time = time.perf_counter() - offline_start

        ff_times_ph, nf_times_ph = [], []
        err_ph_list = []
        mvm_times_ph = []

        for i in range(num_samples):
            if kernel == 'matern':
                param = np.array([param_ell[i], param_nu[i]])
                param_vec = np.array(param, dtype=np.float64).reshape(1, 2)
            else:
                param = np.array([param_ell[i]])
                param_vec = np.array([[param]], dtype=np.float64).reshape(1, 1)

            ff, nf = PH.online_mode(param_vec)
            ff_times_ph.append(ff)
            nf_times_ph.append(nf)

            K_rows = form_kernel_matrix(kernel, random_matrix[idx], random_matrix, param)
            y_true = K_rows @ x0

            t0 = time.perf_counter()
            y_approx = PH.mvm(x0)[idx]
            mvm_times_ph.append(time.perf_counter() - t0)

            rel_err = np.linalg.norm(y_true - y_approx) / np.linalg.norm(y_true)
            err_ph_list.append(rel_err)


        nf_sz, ff_sz = PH.get_size()
        cm_sz = PH.get_coupling_size()
        print(cm_sz)

        offline_data.append({
            "Kernel": kernel,
            "n": sz,
            "Storage (GB)": PH.get_offline_storage() * 8 * (1 / 1024 ** 3),
            "Offline": offline_time,
            "nf time": float(np.mean(nf_times_ph)),
            "ff time": float(np.mean(ff_times_ph)),
        })

        online_data.append({
            "Kernel": kernel,
            "n": sz,
            "cm ratio": cm_sz / (sz * sz),
            "mvm time": float(np.mean(mvm_times_ph)),
            "Error": float(np.mean(err_ph_list)),
            "nf ratio": nf_sz / (sz * sz),
            "ff ratio": ff_sz / (sz * sz),
        })

        PH = None
        Block_T = None
        gc.collect()


eprint("\n===== OFFLINE METRICS TABLE =====")
markdown = markdown_table(offline_data).get_markdown()
eprint(markdown)
eprint("\n===== ONLINE METRICS TABLE =====")
markdown = markdown_table(online_data).get_markdown()
eprint(markdown)
