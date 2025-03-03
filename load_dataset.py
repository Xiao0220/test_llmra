import torch
import numpy as np
import os, pdb
from time import time
import scipy.sparse as sp
import numba as nb
from collections import defaultdict
import random
import argparse
import json
from sklearn.decomposition import PCA

@nb.njit()
def negative_sampling(training_user, training_item, traindata, num_item, num_negative):
    '''
    return: [u,i,j] for training, u interacted with i, not interacted with j
    '''
    trainingData = []
    for k in range(len(training_user)):
        u = training_user[k]
        pos_i = training_item[k]
        for _ in range(num_negative):
            neg_j = random.randint(0, num_item - 1)
            while neg_j in traindata[u]:
                neg_j = random.randint(0, num_item - 1)
            trainingData.append([u, pos_i, neg_j])
    return np.array(trainingData)


@nb.njit()
def Uniform_sampling(batch_users, traindata, num_item):
    trainingData = []
    for u in batch_users:
        pos_items = traindata[u]
        pos_id = np.random.randint(low=0, high=len(pos_items), size=1)[0]
        pos_item = pos_items[pos_id]
        neg_item = random.randint(0, num_item - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, num_item - 1)
        trainingData.append([u, pos_item, neg_item])
    return np.array(trainingData)


class Dataset(object):
    def __init__(self, args):
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data_path = args.data_path
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.num_node = self.num_user + self.num_item
        self.batch_size = args.batch_size
        self.noise_ratio = args.noise_ratio
        ### load and process dataset ###
        self.load_data()
        self.data_to_numba_dict()
        self.training_user, self.training_item = [], []
        self.aug_training_user, self.aug_training_item = [], []
        self.social_i, self.social_j = [], []
        for u, items in self.traindata.items():
            self.training_user.extend([u] * len(items))
            self.training_item.extend(items)
        for u, items in self.aug_traindata.items():
            self.aug_training_user.extend([u] * len(items))
            self.aug_training_item.extend(items)
        for u, users in self.uu_knowledge.items():
            self.social_i.extend([u] * len(users))
            self.social_j.extend(users)
        self.adj_matrix1 = self.lightgcn_adj_matrix()        # original input u-i



    def load_data(self):
        self.traindata = np.load(self.data_path + 'traindata.npy', allow_pickle=True).tolist()
        self.valdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()
        self.testdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()
        self.aug_traindata = np.load(self.data_path + './rel_k/aug_traindata.npy', allow_pickle=True).tolist()
        self.uu_knowledge = np.load(self.data_path + './rel_k/uu_data.npy', allow_pickle=True).tolist()

    #使用 Numba 兼容的数据结构, 提高代码执行速度
    def data_to_numba_dict(self):
        self.traindict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.traindict[key] = np.asarray(list(values))

        self.valdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.valdata.items():
            if len(values) > 0:
                self.valdict[key] = np.asarray(list(values))

        self.testdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.testdata.items():
            if len(values) > 0:
                self.testdict[key] = np.asarray(list(values))

    def _batch_sampling(self, num_negative):
        t1 = time()
        triplet_data = negative_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                         self.traindict, self.num_item, num_negative)
        print('prepare training data cost time:{:.4f}'.format(time() - t1))
        batch_num = int(len(triplet_data) / self.batch_size) + 1
        indexs = np.arange(triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = triplet_data[indexs[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
    def _uniform_sampling(self):
        batch_num = int(len(self.training_user) / self.batch_size) + 1
        for _ in range(batch_num):
            batch_users = random.sample(list(self.traindata.keys()), self.batch_size)
            batch_data = Uniform_sampling(nb.typed.List(batch_users), self.traindict, self.num_item)
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

    def lightgcn_adj_matrix(self):
        '''
        return: sparse adjacent matrix, refer lightgcn
        '''
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        # pdb.set_trace()
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def get_ui_matrix(self):
        user_dim = torch.LongTensor(self.training_user)                     # uid
        item_dim = torch.LongTensor(self.training_item) + self.num_user     # iid

        first_sub = torch.stack([user_dim, item_dim])       # torch.Size([2, 478730])
        second_sub = torch.stack([item_dim, user_dim])      # torch.Size([2, 478730])

        index = torch.cat([first_sub, second_sub], dim=1)        # uu-i: torch.Size([1126610, 2])
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data,
                                            torch.Size([self.num_user + self.num_item, self.num_user + self.num_item]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
            [self.num_user + self.num_item, self.num_user + self.num_item]))
        Graph = Graph.coalesce().to(self.device)
        return Graph, index


    # relation knowledge: delete_i, add_u, add_i
    def get_uu_i_matrix(self):
        user_dim = torch.LongTensor(self.aug_training_user)                     # uid
        item_dim = torch.LongTensor(self.aug_training_item) + self.num_user     # iid
        social_i = torch.LongTensor(self.social_i)                          # uid
        social_j = torch.LongTensor(self.social_j)                          # uid
        first_sub = torch.stack([user_dim, item_dim])       # torch.Size([2, 478730])
        second_sub = torch.stack([item_dim, user_dim])      # torch.Size([2, 478730])
        third_sub = torch.stack([social_i, social_j])       # torch.Size([2, 169150])
        # index = torch.cat([first_sub, second_sub], dim=1)
        # pdb.set_trace()
        index = torch.cat([first_sub, second_sub, third_sub], dim=1)        # uu-i: torch.Size([1126610, 2])
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data,
                                            torch.Size([self.num_user + self.num_item, self.num_user + self.num_item]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
            [self.num_user + self.num_item, self.num_user + self.num_item]))
        Graph = Graph.coalesce().to(self.device)
        # pdb.set_trace()
        social_edge_index = []
        for i in range(index.shape[0]):
            if index[i][0] < self.num_user:
                if index[i][1] < self.num_user:
                    social_edge_index.append(i)
                else:
                    continue
            else:
                continue
        # pdb.set_trace()
        return Graph, social_edge_index

    

