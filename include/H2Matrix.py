import time
import teneva
import numpy as np
from scipy.linalg import khatri_rao
from scipy.linalg.blas import dgemm
from include.LagrangePolynomial import *
from include.Cross import *
from include.FastKron import kron_vec_prod
from include.Kernels import *



class ParamH2Matrix:

    def __init__(self, kernel_name, bc_tree, num_nodes, param_bb, num_param_nodes = 14, ep = 1E-5,
                 translation_invariant = True):
        self.kernel = None
        self._pick_kernel(kernel_name)
        self.ep = ep
        self.data = bc_tree.data
        self.dim = bc_tree.dim
        self.translation_invariant = translation_invariant
        self.param_dim = param_bb.shape[0]
        self.param_bb = param_bb
        self.leaf_lst = bc_tree.leaf_lst
        self.interaction_node_lst = bc_tree.interaction_node_lst
        self.num_nodes = num_nodes
        self.num_param_nodes = num_param_nodes
        self.ff_sz = 0
        self.bc_tree = bc_tree
        self.theta_store = {}
        self.ten_store = {}
        self._form_H2_components(self.bc_tree.left_root)
        self._form_H2_components(self.bc_tree.right_root)


    def get_coupling_size(self):
        cm_sz = 0
        if self.translation_invariant:
            for key, value in self.ten_store.items():
                cm_sz += sum(np.size(arr) for arr in value[:self.dim])
                cm_sz += sum(np.size(arr) for arr in value[self.dim + self.param_dim:])
            for key, value in self.theta_store.items():
                cm_sz += np.size(value)
        else:
            for node in self.interaction_node_lst:
                cm_sz += sum(np.size(arr) for arr in node.tt_ten[:self.dim])
                cm_sz += sum(np.size(arr) for arr in node.tt_ten[self.dim + self.param_dim:])
                cm_sz+= np.size(node.p_mat)
        return cm_sz


    def get_offline_storage(self):
        sz = self.ff_sz
        if self.translation_invariant:
            for key, value in self.ten_store.items():
                sz += sum(np.size(arr) for arr in value)
        else:
            for node in self.interaction_node_lst:
                sz+= sum(np.size(arr) for arr in node.tt_ten)
        for node in self.leaf_lst:
            sz += sum(np.size(arr) for arr in node.tensor)
        return sz

    def get_size(self):
        nf_sz = 0
        ff_sz = self.ff_sz
        for node in self.leaf_lst:
            full = node.full
            nf_sz += np.size(full)
        if self.translation_invariant:
            for key, value in self.ten_store.items():
                ff_sz += sum(np.size(arr) for arr in value[:self.dim])
                ff_sz += sum(np.size(arr) for arr in value[self.dim + self.param_dim:])
            for key, value in self.theta_store.items():
                ff_sz += np.size(value)
        else:
            for node in self.interaction_node_lst:
                ff_sz += sum(np.size(arr) for arr in node.tt_ten[:self.dim])
                ff_sz += sum(np.size(arr) for arr in node.tt_ten[self.dim + self.param_dim:])
                ff_sz += np.size(node.p_mat)
        return nf_sz, ff_sz


    def mvm(self, x):
        y = np.zeros(x.shape)
        self._mat_vec(x, y)
        return y

    def offline_mode(self):
        nf_time = time.perf_counter()
        self._leaf_setup()
        nf_time = time.perf_counter() - nf_time
        ff_time = time.perf_counter()
        self._fill_far_field()
        ff_time = time.perf_counter() - ff_time
        return ff_time, nf_time

    def online_mode(self, param):
        vec_eval = create_basis_matrices(param, self.param_bb, self.num_param_nodes)
        ff_time = time.perf_counter()
        if self.translation_invariant:
            if self.param_dim == 1:
                vector =  np.squeeze(vec_eval[:, :, 0])
                for key, value in self.ten_store.items():
                    param_core = value[self.dim]
                    H = np.einsum('ijk, j -> ik', param_core, vector)
                    self.theta_store[key] = H
            if self.param_dim == 2:
                vector1 = np.squeeze(vec_eval[:, :, 0])
                vector2 = np.squeeze(vec_eval[:, :, 1])
                for key, value in self.ten_store.items():
                    param_core1 = value[self.dim]
                    param_core2 = value[self.dim + 1]
                    H1 = np.einsum('ijk, j -> ik', param_core1, vector1)
                    H2 = np.einsum('ijk, j -> ik', param_core2, vector2)
                    self.theta_store[key] = H1 @ H2
        else:
            if self.param_dim == 1:
                vector =  np.squeeze(vec_eval[:, :, 0])
                for bc in self.interaction_node_lst:
                    param_core = bc.tt_ten[self.dim]
                    H = np.einsum('ijk, j -> ik', param_core, vector)
                    bc.p_mat = H
            if self.param_dim == 2:
                vector1 = np.squeeze(vec_eval[:, :, 0])
                vector2 = np.squeeze(vec_eval[:, :, 1])
                for bc in self.interaction_node_lst:
                    param_core1 = bc.tt_ten[self.dim]
                    param_core2 = bc.tt_ten[self.dim + 1]
                    H1 = np.einsum('ijk, j -> ik', param_core1, vector1)
                    H2 = np.einsum('ijk, j -> ik', param_core2, vector2)
                    bc.p_mat = H1 @ H2
        ff_time = time.perf_counter() - ff_time
        nf_time = time.perf_counter()
        self._form_leaf(param)
        nf_time = time.perf_counter() - nf_time
        return ff_time, nf_time


    def _mat_vec(self, x, y):
        self._fast_forward(self.bc_tree.right_root, x)
        self._interaction_mult(y, x)
        self._fast_backward(self.bc_tree.left_root, y)


    def _interaction_mult(self, y, x):
        for node in self.interaction_node_lst:
            if node.is_admissible:
                if self.translation_invariant:
                    key = (node.right_cluster.level, node.diff)
                    p_mat = self.theta_store[key]
                    tt_ten = self.ten_store[key]
                else:
                    p_mat = node.p_mat
                    tt_ten = node.tt_ten
                v0 = (self._int_mult_vec(tt_ten, p_mat, node.right_cluster.v_hat))
                if node.left_cluster.v_hat is not None:
                    node.left_cluster.v_hat = (node.left_cluster.v_hat + v0)
                else:
                    node.left_cluster.v_hat = v0
        for node in self.leaf_lst:
            y[node.left_cluster.index_set] = (y[node.left_cluster.index_set] +
                                              (node.full @ x[node.right_cluster.index_set]))


    def _fast_forward(self, tree_cluster, vec):
        if tree_cluster.is_leaf():
            tree_cluster.v_hat = (np.transpose(self.chain_face_split(tree_cluster.V_lst[::-1]))
                                  @ vec[tree_cluster.index_set])
        else:
            tree_cluster.v_hat = None
            for son in tree_cluster.sons:
                self._fast_forward(son, vec)
                if type(tree_cluster.v_hat) != type(None):
                    tree_cluster.v_hat = (tree_cluster.v_hat + kron_vec_prod(son.B_lst[::-1],
                                                                             son.v_hat, transpose=True))
                else:
                    tree_cluster.v_hat = kron_vec_prod(son.B_lst[::-1],
                                                       son.v_hat, transpose=True)


    def _fast_backward(self, tree_cluster, y):
        if tree_cluster.is_leaf():
            y[tree_cluster.index_set] = (y[tree_cluster.index_set] +
                                         ( np.squeeze((self.chain_face_split(tree_cluster.V_lst[::-1]) @
                                           tree_cluster.v_hat))))
        else:
            if tree_cluster.v_hat is not None:
                for son in tree_cluster.sons:
                    if son.v_hat is not None:
                        son.v_hat = (son.v_hat + kron_vec_prod(son.B_lst[::-1], tree_cluster.v_hat))
                    else:
                        son.v_hat = kron_vec_prod(son.B_lst[::-1], tree_cluster.v_hat)
                    self._fast_backward(son, y)
            else:
                for son in tree_cluster.sons:
                    self._fast_backward(son, y)
        tree_cluster.v_hat = None

    def _form_H2_components(self, node):
        # form the cluster basis
        if node.is_leaf():
            V_mats = create_basis_matrices(self.data[node.index_set, :], node.bb_box, self.num_nodes)
            self.ff_sz += np.size(V_mats)
            node.V_lst = [V_mats[:, :, i] for i in range(V_mats.shape[-1])]
        else:
            for son in node.sons:
                self._form_H2_components(son)
            # now form the parent transfer matrices using the sons
            for son in node.sons:
                points = create_node_map_mats(son.bb_box, self.num_nodes).T
                B_mats = create_basis_matrices(points, node.bb_box, self.num_nodes)
                self.ff_sz += np.size(B_mats)
                son.B_lst = [B_mats[:, :, i] for i in range(B_mats.shape[-1])]

    def _leaf_setup(self):
        param_c_matrix  = create_node_map_mats(self.param_bb, self.num_param_nodes)
        for node in self.leaf_lst:
            fun = lambda M: self.nf_map(M, self.data[node.left_idx, :], self.data[node.right_idx, :], param_c_matrix,
                                   self.kernel )
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
                node.full = (( U @ (V @ arr.T)).reshape([np.size(node.left_idx),
                                                                       np.size(node.right_idx)]))
        if self.param_dim == 2:
            arr_1 = np.squeeze(vec_eval[:, :, 0])
            arr_2 = vec_eval[:, :, 1]
            for i in range(len(self.leaf_lst)):
                node = self.leaf_lst[i]
                U, V, W = node.tensor
                H1 = np.einsum('ijk, j -> ik', V, arr_1)
                H2 = W @ arr_2.T
                tmp = H1 @ H2
                node.full = ((U @ tmp).reshape([np.size(node.left_idx),
                                                              np.size(node.right_idx)]))

    def _fill_far_field(self):
        for node in self.interaction_node_lst:
            source_bb = node.left_cluster.bb_box
            target_bb = node.right_cluster.bb_box
            if self.translation_invariant:
                if (node.left_cluster.level, node.diff) not in self.ten_store:
                    tt_ten = self._form_coupling_tt(source_bb, target_bb)
                    self.ten_store[(node.left_cluster.level, node.diff)] = tt_ten
                    self.theta_store[(node.left_cluster.level, node.diff)] = None
            else:
                node.tt_ten = self._form_coupling_tt(source_bb, target_bb)

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

    def _form_coupling_tt(self, source_bb, target_bb):
        source_c_matrix = create_node_map_mats(source_bb, self.num_nodes)
        target_c_matrix = create_node_map_mats(target_bb, self.num_nodes)
        param_c_matrix  = create_node_map_mats(self.param_bb, self.num_param_nodes)
        s = lambda M: vec_kernel(M, self.kernel, source_c_matrix, target_c_matrix, param_c_matrix)
        tt_cores = self._form_interaction_tt(s, self.dim, self.param_dim, self.num_param_nodes, self.num_nodes, self.ep)
        tt_cores = self.ensure_F_cores(tt_cores)
        return tt_cores

    def _int_mult_vec(self, cores, param_mat, vec):
        H = np.asfortranarray(param_mat)
        vec = np.asfortranarray(vec).reshape(-1, 1, order = 'F')
        back_cores = cores[:self.dim]
        front_cores = cores[self.dim + self.param_dim:]

        # Begin Algorithm 4.4 from (*insert paper here*)
        for c in front_cores[::-1]:
            c = self.flatten_last_two_dims(c)
            a, b = np.shape(c)
            vec = vec.reshape([int(np.size(vec) / b), b], order='F') @ c.T
        vec = H @ vec.T
        for c in back_cores[::-1]:
            c = self.flatten_first_two_dims(c)
            a, b = np.shape(c)
            vec = c @ vec.reshape([b, int(np.size(vec) / b)], order='F')
        # End Algorithm 4.4 from (*insert paper here)

        return vec.reshape(-1, 1, order='F')

    @staticmethod
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

    @staticmethod
    def _form_interaction_tt(my_vec_kernel, s_dim, p_dim, p_num_nodes, num_nodes, ep):
        u = [num_nodes]*s_dim + [p_num_nodes]*p_dim + [num_nodes]*s_dim
        while True:
            cores = greedy_cross(u, my_vec_kernel, ep, 1000)
            if not any(np.isnan(core).any() for core in cores):
                break  # exit if all cores are valid
        cores = teneva.truncate(cores, ep/2)
        return cores


    @staticmethod
    def flatten_first_two_dims(t):
        a, b, c = t.shape
        return t.reshape(a * b, c, order='F')

    @staticmethod
    def flatten_last_two_dims(t):
        a, b, c = t.shape
        return t.reshape(a, b * c, order='F')

    @staticmethod
    def chain_face_split(matrices):
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.transpose(khatri_rao(result.T, matrix.T))
        return result
    @staticmethod
    # Make sure each core is fortran contiguous so the reshapes, in the MVM operation, can be done in O(1) time.
    def ensure_F_cores(cores, dtype=np.float64):
        return tuple(np.require(c, dtype=dtype, requirements=['F', 'ALIGNED', 'WRITEABLE'])
                     for c in cores)






class H2Matrix:
    def __init__(self, kernel_name, bc_tree, num_nodes, translation_invariant=False, ep=1E-3):
        self.kernel1 = None
        self.kernel2 = None
        self._pick_kernel(kernel_name)
        self.leaf_lst = bc_tree.leaf_lst
        self.interaction_node_lst = bc_tree.interaction_node_lst
        self.bb_tree = bc_tree
        self.data = bc_tree.data
        self.ep = ep
        self.num_nodes = num_nodes
        self.ff_sz = 0
        self.left_root = bc_tree.left_root
        self.right_root = bc_tree.right_root
        self.translation_invariant = translation_invariant
        self.dim = bc_tree.dim
        self.bb_store = {}
        self.coupling_store = {}
        self._offline_mode()

    def get_coupling_size(self):
        cm_sz = 0
        if self.translation_invariant:
            for key, value in self.coupling_store.items():
                cm_sz += np.size(value[0]) + np.size(value[1])
        else:
            for node in self.interaction_node_lst:
                cm_sz += np.size(node.coupling_tensor[0]) + np.size(node.coupling_tensor[1])
        return cm_sz

    def get_size(self):
        nf_sz = 0
        ff_sz = self.ff_sz
        for node in self.leaf_lst:
            full = node.full
            nf_sz += np.size(full)
        if self.translation_invariant:
            for key, value in self.coupling_store.items():
                ff_sz += np.size(value[0]) + np.size(value[1])
        else:
            for node in self.interaction_node_lst:
                ff_sz += np.size(node.coupling_tensor[0]) + np.size(node.coupling_tensor[1])
        return nf_sz, ff_sz

    def online_mode(self, param):
        ff_time = time.perf_counter()
        if self.translation_invariant:
            for key, value in self.bb_store.items():
                self.coupling_store[key] = self._form_coupling_tt(value[0],  value[1], param)
        else:
            for node in self.interaction_node_lst:
                value = (node.left_cluster.bb_box, node.right_cluster.bb_box)
                node.coupling_tensor = self._form_coupling_tt(value[0],  value[1], param)
        ff_time = time.perf_counter() - ff_time
        nf_time = time.perf_counter()
        self._fill_near_field(param)
        nf_time = time.perf_counter() - nf_time
        return nf_time, ff_time

    def mvm(self, x):
        y = np.zeros(x.shape)
        self._mat_vec(x, y)
        return y

    def _fill_near_field(self, param):
        for node in self.leaf_lst:
            left = self.data[node.left_cluster.index_set, :]
            right = self.data[node.right_cluster.index_set, :]
            node.full = self.kernel(left, right, *param, pointwise=False)

    def _offline_mode(self):
        if self.translation_invariant:
            for node in self.interaction_node_lst:
                key = (node.right_cluster.level, node.diff)
                self.bb_store[key] = (node.left_cluster.bb_box, node.right_cluster.bb_box)
        self._form_H2_components(self.left_root)
        self._form_H2_components(self.right_root)

    def _mat_vec(self, x, y):
        self._fast_forward(self.right_root, x)
        self._interaction_mult(y, x)
        self._fast_backward(self.left_root, y)


    def _interaction_mult(self, y, x):
        for node in self.interaction_node_lst:
            if node.is_admissible:
                if self.translation_invariant:
                    key = (node.right_cluster.level, node.diff)
                    M1, M2 = self.coupling_store[key]
                    t0 = time.perf_counter()
                    v0 = M1 @ (M2 @ node.right_cluster.v_hat)
                    t1 = time.perf_counter() - t0
                    v0 = v0.reshape(-1, 1)
                else:
                    M1, M2 = node.coupling_tensor
                    v0 = M1 @ (M2 @ node.right_cluster.v_hat)
                    v0 = v0.reshape(-1, 1)

                if node.left_cluster.v_hat is not None:
                    node.left_cluster.v_hat = (node.left_cluster.v_hat + v0)
                else:
                    node.left_cluster.v_hat = v0

        for node in self.leaf_lst:
            y[node.left_cluster.index_set] = (y[node.left_cluster.index_set] +
                                                            (node.full @
                                                             x[node.right_cluster.index_set]))

    def _fast_forward(self, tree_cluster, vec):
        if tree_cluster.is_leaf():
            tree_cluster.v_hat = (np.transpose(self.chain_face_split(tree_cluster.V_lst[::-1]))
                                  @ vec[tree_cluster.index_set])
        else:
            tree_cluster.v_hat = None
            for son in tree_cluster.sons:
                self._fast_forward(son, vec)
                if type(tree_cluster.v_hat) != type(None):
                    tree_cluster.v_hat = (tree_cluster.v_hat + kron_vec_prod(son.B_lst[::-1],
                                                                             son.v_hat, transpose=True))
                else:
                    tree_cluster.v_hat = kron_vec_prod(son.B_lst[::-1],
                                                       son.v_hat, transpose=True)


    def _fast_backward(self, tree_cluster, y):
        if tree_cluster.is_leaf():
            y[tree_cluster.index_set] = (y[tree_cluster.index_set] +
                                         (np.squeeze((self.chain_face_split(tree_cluster.V_lst[::-1]) @
                                                      tree_cluster.v_hat))))
        else:
            if tree_cluster.v_hat is not None:
                for son in tree_cluster.sons:
                    if son.v_hat is not None:
                        son.v_hat = (son.v_hat + kron_vec_prod(son.B_lst[::-1], tree_cluster.v_hat))
                    else:
                        son.v_hat = kron_vec_prod(son.B_lst[::-1], tree_cluster.v_hat)
                    self._fast_backward(son, y)
            else:
                for son in tree_cluster.sons:
                    self._fast_backward(son, y)
        tree_cluster.v_hat = None

    def _form_H2_components(self, node):
        # form the cluster basis
        if node.is_leaf():
            V_mats = create_basis_matrices(self.data[node.index_set, :], node.bb_box, self.num_nodes)
            self.ff_sz += np.size(V_mats)
            node.V_lst = [V_mats[:, :, i] for i in range(V_mats.shape[-1])]
        else:
            for son in node.sons:
                self._form_H2_components(son)
            # now form the parent transfer matrices using the sons
            for son in node.sons:
                points = create_node_map_mats(son.bb_box, self.num_nodes).T
                B_mats = create_basis_matrices(points, node.bb_box, self.num_nodes)
                self.ff_sz += np.size(B_mats)
                son.B_lst = [B_mats[:, :, i] for i in range(B_mats.shape[-1])]

    def _form_coupling_tt(self, source_bb, target_bb, param):
        source_c_matrix = create_node_map_mats(source_bb, self.num_nodes)
        target_c_matrix = create_node_map_mats(target_bb, self.num_nodes)
        X_1d = [source_c_matrix[i, :] for i in range(self.dim)]
        Y_1d = [target_c_matrix[i, :] for i in range(self.dim)]
        X_grid = np.stack(np.meshgrid(*X_1d, indexing='ij'), axis=-1).reshape(-1, self.dim, order='F')
        Y_grid = np.stack(np.meshgrid(*Y_1d, indexing='ij'), axis=-1).reshape(-1, self.dim, order='F')
        my_kernel = lambda X, Y: self.kernel(X, Y, *param)
        return aca_partial(X_grid, Y_grid, self.ep, my_kernel)

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
    def chain_face_split(matrices):
        result = matrices[0]
        for matrix in matrices[1:]:
            result = H2Matrix.face_split(result, matrix)
        return result

    @staticmethod
    def face_split(A, B):
        return np.transpose(khatri_rao(A.T, B.T))
