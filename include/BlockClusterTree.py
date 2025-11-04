import numpy as np
from include.ClusterTree import *


class BlockCluster:
    def __init__(self, left_cluster, right_cluster):
        # For \sigma \times \tau \in \mc{T}_{I \times I}, \sigma corresponds to left_cluster and \tau corresponds to
        # right cluster.
        self.left_cluster = left_cluster
        self.right_cluster = right_cluster
        self.left_idx = left_cluster.index_set
        self.right_idx = right_cluster.index_set
        self.level = None
        self.sons = []
        self.U = None
        self.V = None
        self.coupling_tensor = None
        self.full = None
        self.diff = None
        self.is_admissible = False

    def truncate(self):
        self.level = self.left_cluster.level
        self.left_cluster = None
        self.right_cluster = None


class BlockTree:
    def __init__(self, left_tree, right_tree, data, adm_param):
        self.left_root = left_tree.root
        self.right_root = right_tree.root
        self.data = data
        self.dim = self.data.shape[1]
        self.adm_param = adm_param
        self.interaction_node_lst = []
        self.leaf_lst = []
        self.root = BlockCluster(self.left_root, self.right_root)
        self._build_block_tree(self.root)

    def validate(self):
        sz = 0
        for node in self.interaction_node_lst:
            sz += np.size(node.left_cluster.index_set)*np.size(node.right_cluster.index_set)
        for node in self.leaf_lst:
            sz += np.size(node.left_cluster.index_set)*np.size(node.right_cluster.index_set)
        if sz == self.data.shape[0]**2:
            return True
        else:
            return False

    def _build_block_tree(self, node):
        left = node.left_cluster
        right = node.right_cluster
        admissible = self._isadmissible(left.bb_box, right.bb_box, self.adm_param)

        if not admissible and not(left.is_leaf()) and not(right.is_leaf()):
            for left_son in left.sons:
                for right_son in right.sons:
                    node.sons.append(BlockCluster(left_son, right_son))
            for son in node.sons:
                self._build_block_tree(son)
        else:
            node.is_admissible = admissible
            node.sons = None
            if admissible:
                interval_diff = tuple(
                np.array(right.m, dtype=int) - np.array(left.m, dtype=int))
                node.diff = interval_diff
                self.interaction_node_lst.append(node)
            else:
                self.leaf_lst.append(node)

    @staticmethod
    def _diam(bounding_box):
        return np.sqrt(np.sum(np.square(bounding_box[:, 1] - bounding_box[:, 0])))

    @staticmethod
    def _dist(bounding_box_1, bounding_box_2):
        max_with_zero_1 = np.sum(np.square(np.maximum(bounding_box_1[:, 0] - bounding_box_2[:, 1], 0)))
        max_with_zero_2 = np.sum(np.square(np.maximum(bounding_box_2[:, 0] - bounding_box_1[:, 1], 0)))
        return np.sqrt(max_with_zero_1 + max_with_zero_2)

    @staticmethod
    def _isadmissible(left_bb, right_bb, adm_param):
        return (max(BlockTree._diam(left_bb), BlockTree._diam(right_bb)) <=
                adm_param * BlockTree._dist(left_bb, right_bb))
