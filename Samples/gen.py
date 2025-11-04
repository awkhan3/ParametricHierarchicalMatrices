import numpy as np

dim = 3
sizes = {
    "small": 8**4,
    "medium": 8**5,
    "large": 8**6,
    "err": 3*8**5
}

num_samples = 30
high_ell = 1.0
param_ell = np.random.uniform(.25, high_ell, num_samples)
param_ell_extended = np.random.uniform(.1, high_ell, num_samples)
param_nu  = np.random.uniform(0.5, 3, num_samples)

np.save("param_ell.npy", param_ell)
np.save("param_ell_extended.npy", param_ell_extended)
np.save("param_nu.npy", param_nu)

for label, sz in sizes.items():
    matrix = np.random.rand(sz, dim)
    np.save(f"{label}_matrix.npy", matrix)
    idx = np.random.choice(sz, 200, replace=False)
    np.save(f"{label}_idx.npy", idx)
    x0 = np.random.rand(sz)
    np.save(f"{label}_x0.npy", x0)
