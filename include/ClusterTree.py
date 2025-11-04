import numpy as np
from numba import jit
from itertools import product


class Cluster:
    def __init__(self, index_set):
        self.index_set = index_set
        self.sons = []
        self.level = 0
        self.m = None
        self.bb_box = None
        # Attributes for the H2-matrix.
        self.V_lst = None
        self.B_lst = None
        self.v_hat = None
    def is_leaf(self):
        return len(self.sons) == 0


class ClusterTree:
    def __init__(self, data, lvl_max = 1):
        self.data = data
        self.lvl_max = lvl_max
        self.index_set = list(np.arange(data.shape[0]))
        self.dimension = data.shape[1]
        self.ref_box = np.array([[0, 1]] * self.dimension)
        self.root = Cluster(self.index_set)
        self.root.m = [0] * self.dimension
        self.root.bb_box =  self.ref_box
        self.bin_strings = list(product([0, 1], repeat=self.dimension))
        self._construct_tree(self.root)

    # The domain is B = [0, 1]^{dimension}. Consequently, at a fixed level l > 0, every \sigma \in \mc{T}_{I} such
    # that $level(\sigma) = l$ can be represented in the form B_{\sigma} = \alpha*(R_{l} +  m),
    # where R_{l} = [0, 1]^{dimension}, alpha = 1/2^{dl}, and m \in \mathbb{N}^{dimension}.
    # This partitioning idea is from https://www.sciencedirect.com/science/article/pii/S0955799725000785
    def _construct_tree(self, cluster, level = 1):
        if level > self.lvl_max:
            return
        # Each binary string is a coordinate representing a block of the partition. For example, [0, 1]^{2} is
        # partitioned into [0,.5]^{2}, [0,.5] \times [.5, 1], [.5, 1] \times [0, .5], and [.5, 1]^{2}, and each block
        # maps to the binary strings (0,0), (0, 1), (1,0), and (1,1), respectively.
        for str in self.bin_strings:
            str = np.array(str)
            m = np.copy(cluster.m)

            # The values m and alpha must be updated, since at each level the coordinates are halved w.r.t each axis.
            mult = 2 if level > 1 else 1
            m = mult*m
            alpha = 1/(2**level)

            m = np.add(m, str)
            m_vec = np.array(m, dtype=int)
            shift = m_vec[:, None] * alpha
            shifted_box = self.ref_box * alpha + shift
            # Now, we can prune the index set.
            pruned_index_set = self._prune_index_set(self.data, np.array(cluster.index_set), shifted_box)
            if len(pruned_index_set) > 0:
                new_cluster = Cluster(pruned_index_set)
                new_cluster.m = m
                new_cluster.level = level
                new_cluster.bb_box = shifted_box
                cluster.sons.append(new_cluster)

        for son in cluster.sons:
            self._construct_tree(son, level + 1)


    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def _prune_index_set(data, index_set, shifted_box):
        dim = data.shape[1]
        pruned = []
        for index in index_set:
            inside = True
            for d in range(dim):
                val = data[index, d]
                if val < shifted_box[d, 0] or val > shifted_box[d, 1]:
                    inside = False
                    break # The element Data[index, :] is not in this box.
            if inside:
                pruned.append(index)
        return np.array(pruned)





