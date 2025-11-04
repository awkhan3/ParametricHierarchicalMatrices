import time
import teneva
from functools import partial
from include.Cross import greedy_cross
from include.Cross import aca_partial
from include.LagrangePolynomial import *
from include.Kernels import *
import gc


class ParamHMatrix:

    def __init__(self, kernel_name, bc_tree, num_nodes, param_bb, param_nodes=14, ep=1E-5):
        self.kernel = None
        self._pick_kernel(kernel_name)
        self.num_nodes = num_nodes
        self.num_param_nodes = param_nodes
        self.ep = ep
        self.dim = bc_tree.dim
        self.data = bc_tree.data
        self.leaf_lst = bc_tree.leaf_lst
        self.interaction_node_lst = bc_tree.interaction_node_lst
        self.param_bb = param_bb
        self.param_dim = param_bb.shape[0]
        self.theta_store = {}
        self.ten_store = {}
        self.truncate()

    def truncate(self):
        for node in self.interaction_node_lst + self.leaf_lst:
            node.level = node.left_cluster.level
            node.left_idx = node.left_cluster.index_set
            node.right_idx = node.right_cluster.index_set
            node.left_bb = node.left_cluster.bb_box
            node.right_bb = node.right_cluster.bb_box
            node.truncate()
        gc.collect()

    def mvm(self, x):
        y = np.zeros(np.size(x))
        for i in range(len(self.leaf_lst)):
            node = self.leaf_lst[i]
            full = node.full
            y[node.left_idx] += full @ x[node.right_idx]
        for i in range(len(self.interaction_node_lst)):
            node = self.interaction_node_lst[i]
            H = self.theta_store[(node.level, node.diff)]
            U = node.U
            V = node.V
            temp = V @ x[node.right_idx]
            temp = H @ temp
            y[node.left_idx] += U @ temp
        return y

    def get_offline_storage(self):
        sz = 0
        for key, value in self.ten_store.items():
            sz += sum(np.size(arr) for arr in value)
        for node in self.leaf_lst:
            sz += sum(np.size(arr) for arr in node.tensor)
        for node in self.interaction_node_lst:
            sz += np.size(node.U) + np.size(node.V)
        return sz

    def get_size(self):
        nf_sz = 0
        ff_sz = 0
        for node in self.leaf_lst:
            full = node.full
            nf_sz += np.size(full)
        for node in self.interaction_node_lst:
            ff_sz += np.size(node.U) + np.size(node.V)
        for key, value in self.theta_store.items():
            ff_sz += np.size(value)
        return nf_sz, ff_sz

    def get_mean_rank(self):
        L = []
        for key, value in self.theta_store.items():
            L.append(np.max(np.shape(value)))
        return np.mean(L)

    def online_mode(self, param):
        vec_eval = create_basis_matrices(param, self.param_bb, self.num_param_nodes)
        ff_time = time.perf_counter()
        if self.param_dim == 1:
            vector = np.squeeze(vec_eval[:, :, 0])
            for key, value in self.ten_store.items():
                param_core = value
                H = np.einsum('ijk, j -> ik', param_core, vector)
                self.theta_store[key] = H
        if self.param_dim == 2:
            vector1 = np.squeeze(vec_eval[:, :, 0])
            vector2 = np.squeeze(vec_eval[:, :, 1])
            for key, value in self.ten_store.items():
                param_core1 = value[0]
                param_core2 = value[1]
                H1 = np.einsum('ijk, j -> ik', param_core1, vector1)
                H2 = np.einsum('ijk, j -> ik', param_core2, vector2)
                self.theta_store[key] = H1 @ H2
        ff_time = time.perf_counter() - ff_time

        nf_time = time.perf_counter()
        self._form_leaf(param)
        nf_time = time.perf_counter() - nf_time
        return ff_time, nf_time

    def offline_mode(self):
        nf_time = time.perf_counter()
        self._leaf_setup()
        nf_time = time.perf_counter() - nf_time
        ff_time = time.perf_counter()
        self._fill_far_field()
        for key, value in self.ten_store.items():
            if self.param_dim == 1:
                self.ten_store[key] = value[self.dim]
            elif self.param_dim == 2:
                self.ten_store[key] = [value[self.dim], value[self.dim + 1]]
        ff_time = time.perf_counter() - ff_time
        return ff_time, nf_time

    def _fill_far_field(self):
        for node in self.interaction_node_lst:
            X = self.data[node.left_idx, :]
            Y = self.data[node.right_idx, :]
            source_bb = node.left_bb
            target_bb = node.right_bb
            U_mats = create_basis_matrices(X, source_bb, self.num_nodes)
            V_mats = create_basis_matrices(Y, target_bb, self.num_nodes)

            if (node.level, node.diff) not in self.ten_store:
                tt_ten = self._form_coupling_tt(source_bb, target_bb)
                self.ten_store[(node.level, node.diff)] = tt_ten
                self.theta_store[(node.level, node.diff)] = None
            else:
                tt_ten = self.ten_store[(node.level, node.diff)]

            tt_factors = tt_ten[:self.dim] + tt_ten[-self.dim:]

            for i in range(self.dim):
                s1, s2, s3 = tt_factors[i].shape
                tt_factors[i] = np.reshape(tt_factors[i], (s1 * s2, s3), order='F')

            for i in range(self.dim, 2 * self.dim):
                s1, s2, s3 = tt_factors[i].shape
                tt_factors[i] = np.reshape(tt_factors[i], (s1, s2 * s3), order='F')

            u = U_mats[:, :, 0] @ tt_factors[0]
            for t in range(1, self.dim):
                u = fast_dot_mult_1(U_mats[:, :, t], u, tt_factors[t])

            tt_factors = tt_factors[self.dim:]
            tt_factors.reverse()

            v = tt_factors[0] @ V_mats[:, :, -1].T
            for t in range(1, self.dim):
                v = fast_dot_mult_2(v, V_mats[:, :, -1 - t].T, tt_factors[t])
            node.U = np.squeeze(u)
            node.V = np.squeeze(v)

    def _form_coupling_tt(self, source_bb, target_bb):
        source_c_matrix = create_node_map_mats(source_bb, self.num_nodes)
        target_c_matrix = create_node_map_mats(target_bb, self.num_nodes)
        param_c_matrix = create_node_map_mats(self.param_bb, self.num_param_nodes)
        s = lambda M: vec_kernel(M, self.kernel, source_c_matrix, target_c_matrix, param_c_matrix)
        return self._form_interaction_tt(s, self.dim, self.param_dim, self.num_nodes, self.num_param_nodes, self.ep)

    def _leaf_setup(self):
        param_c_matrix = create_node_map_mats(self.param_bb, self.num_param_nodes)
        for node in self.leaf_lst:
            fun = lambda M: nf_map(M, self.data[node.left_idx, :], self.data[node.right_idx, :], param_c_matrix,
                                   self.kernel)
            u = [len(node.left_idx)*len(node.right_idx)] + [self.num_param_nodes]*self.param_dim

            while True:
                T = greedy_cross(u, fun, self.ep, 1000)
                if not any(np.isnan(core).any() for core in T):
                    break  # exit if all cores are valid

            cores = teneva.truncate(T, self.ep)
            cores[0] = np.squeeze(cores[0])
            cores[-1] = np.squeeze(cores[-1])
            node.tensor = cores

    def _form_leaf(self, param):
        vec_eval = create_basis_matrices(param, self.param_bb, self.num_param_nodes)
        if self.param_dim == 1:
            arr = vec_eval[:, :, 0]
            for i in range(len(self.leaf_lst)):
                node = self.leaf_lst[i]
                U, V = node.tensor
                node.full = ((U @ (V @ arr.T)).reshape([np.size(node.left_idx),
                                                        np.size(node.right_idx)]))
        if self.param_dim == 2:
            arr_1 = vec_eval[:, :, 0]
            arr_2 = vec_eval[:, :, 1]
            for i in range(len(self.leaf_lst)):
                node = self.leaf_lst[i]
                U, V, W = node.tensor
                H1 = np.tensordot(V, arr_1.T, axes=([1], [0]))
                H2 = W @ arr_2.T
                tmp = np.squeeze(H1) @ H2
                node.full = ((U @ tmp).reshape([np.size(node.left_idx),
                                                np.size(node.right_idx)]))

    def _pick_kernel(self, kernel_name):
        match kernel_name:
            case "exponential":
                self.kernel = n_exponential
            case "matern":
                self.kernel = n_matern
            case "thin_plate_spline":
                self.kernel = n_thin_plate_spline
            case "multiquadric":
                self.kernel = n_multiquadrics
            case "squared_exponential":
                self.kernel = n_squared_exponential

    @staticmethod
    def _form_interaction_tt(my_vec_kernel, s_dim, p_dim, num_nodes, p_num_nodes, ep):
        u = [num_nodes]*int(s_dim) + [p_num_nodes]*int(p_dim) + [num_nodes]*int(s_dim)
        while True:
            cores = greedy_cross(u, my_vec_kernel, ep, 1000)
            if not any(np.isnan(core).any() for core in cores):
                break  # exit if all cores are valid
        return teneva.truncate(cores, ep / 2)


@jit(nopython=True, fastmath=True, cache=True)
def fast_dot_mult_1(A, B, C):
    R = np.zeros((A.shape[0], C.shape[1]))
    for i in range(A.shape[0]):
        R[i, :] = np.outer(A[i, :], B[i, :]).reshape(1, A.shape[1] * B.shape[1]) @ C
    return R


@jit(nopython=True, fastmath=True, cache=True)
def fast_dot_mult_2(A, B, C):
    R = np.zeros((C.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        R[:, i] = C @ np.outer(A[:, i], B[:, i]).reshape(A.shape[0] * B.shape[0])
    return R


class HMatrix:

    def __init__(self, kernel_name, bc_tree, ep=1E-5):
        self.kernel = None
        self._pick_kernel(kernel_name)
        self.ep = ep
        self.data = bc_tree.data
        self.leaf_lst = bc_tree.leaf_lst
        self.interaction_node_lst = bc_tree.interaction_node_lst
        self.truncate()

    def truncate(self):
        for node in self.interaction_node_lst + self.leaf_lst:
            node.truncate()
        gc.collect()

    def mvm(self, x):
        y = np.zeros(np.size(x))
        for i in range(len(self.leaf_lst)):
            node = self.leaf_lst[i]
            y[node.left_idx] += node.full @ x[node.right_idx]
        for i in range(len(self.interaction_node_lst)):
            node = self.interaction_node_lst[i]
            temp = node.V @ x[node.right_idx]
            y[node.left_idx] += node.U @ temp
        return y

    def get_size(self):
        return 0, 0

    def get_mean_rank(self):
        L = []
        for node in self.interaction_node_lst:
            U = node.U
            V = node.V
            L.append(np.max([U.shape[1], V.shape[0]]))
        return np.mean(L)

    def online_mode(self, param):
        ff_time = time.perf_counter()
        self._fill_far_field(param)
        ff_time = time.perf_counter() - ff_time
        nf_time = time.perf_counter()
        self._fill_near_field(param)
        nf_time = time.perf_counter() - nf_time
        return ff_time, nf_time

    def _fill_near_field(self, param):
        for node in self.leaf_lst:
            source_points = self.data[node.left_idx, :]
            target_points = self.data[node.right_idx, :]
            node.full = self.kernel(source_points, target_points, *param, pointwise=False)

    def _fill_far_field(self, param):
        for node in self.interaction_node_lst:
            my_kernel = lambda X, Y: self.kernel(X, Y, *param)
            U, V = aca_partial(self.data[node.left_idx, :], self.data[node.right_idx, :], self.ep, my_kernel)
            node.U = U
            node.V = V

    def _pick_kernel(self, kernel_name):
        match kernel_name:
            case "exponential":
                self.kernel = n_exponential
            case "matern":
                self.kernel = n_matern
            case "thin_plate_spline":
                self.kernel = n_thin_plate_spline
            case "multiquadric":
                self.kernel = n_multiquadrics
            case "squared_exponential":
                self.kernel = n_squared_exponential

