import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pdb
init = nn.init.xavier_uniform_
def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class LLaRD(nn.Module):
    def __init__(self, args, dataset):
        super(LLaRD, self).__init__()
        self.args = args
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.model = args.model
        self.gcn_layer = args.gcn_layer
        self.latent_dim = args.latent_dim
        self.embedding_size = args.latent_dim
        self.init_type = args.init_type
        self.l2_reg = args.l2_reg
        self.beta = args.beta
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.delta = 0.01
        self.is_training = False
        self.keep_rate = args.keep_rate
        self.edge_bias = args.edge_bias
        self.batch_size = args.batch_size
        self.prf_weight = args.prf_weight
        self.str_weight = args.str_weight
        self.adj_matrix, self.cf_index = dataset.get_ui_matrix()
        self.aug_adj_matrix, self.aug_cf_index = dataset.get_uu_i_matrix()
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.kd_temperature = args.kd_temperature
        self.user_prf, self.item_prf = self.load_embedding()
        self.mlp = nn.Sequential(
            nn.Linear(self.user_prf.shape[1], (self.user_prf.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.user_prf.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self._init_weights()

    def load_embedding(self):
        predir = self.args.data_path
        path_user = predir + 'prf_k/usr_emb_np.pkl'
        path_item = predir + 'prf_k/itm_emb_np.pkl'
        with open(path_user, 'rb') as f:
            user_prf = torch.tensor(pickle.load(f)).float().cuda()
        with open(path_item, 'rb') as f:
            item_prf = torch.tensor(pickle.load(f)).float().cuda()
        return user_prf, item_prf

    def _init_weights(self):
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        self.user_str_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_str_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        if self.init_type == 'norm':
            nn.init.normal_(self.user_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_embeddings.weight, std=0.01)
            nn.init.normal_(self.user_str_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_str_embeddings.weight, std=0.01)
        elif self.init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embeddings.weight)
            nn.init.xavier_normal_(self.item_embeddings.weight)
            nn.init.xavier_normal_(self.user_str_embeddings.weight, std=0.01)
            nn.init.xavier_normal_(self.item_str_embeddings.weight, std=0.01)
        else:
            raise NotImplementedError
        # graph learner
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=2*self.latent_dim, out_features=self.latent_dim, bias=True)
        self.linear_2 = nn.Linear(in_features=self.latent_dim, out_features=1, bias=True)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        return None


    def graph_learner(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        row, col = self.cf_index[:,0], self.cf_index[:,1]
        row_emb = all_emb[row]
        col_emb = all_emb[col]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).to(self.device)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias  # self.edge_bias
        # weights = torch.ones_like(self.adj_matrix.values())
        # weights[self.social_index] = mask_gate_input
        # weights = weights.detach()
        masked_Graph = torch.sparse.FloatTensor(self.adj_matrix.indices(), self.adj_matrix.values()*mask_gate_input, torch.Size(
            [self.num_user + self.num_item, self.num_user + self.num_item]))
        masked_Graph = masked_Graph.coalesce().to(self.device)
        return masked_Graph

    def edge_dropper(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask]# / keep_rate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

    def forward(self, adj_matrix, g_type='cf', kr=1.0):
        if g_type == 'cf':
            ego_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        elif g_type == 'llm':
            ego_emb = torch.cat([self.user_str_embeddings.weight, self.item_str_embeddings.weight], dim=0)
     
        if self.model == 'GMF':
            user_emb, item_emb = torch.split(ego_emb, [self.num_user, self.num_item])
        elif self.model == 'LGN':
            all_emb = [ego_emb]
            for _ in range(self.gcn_layer):
                tmp_emb = torch.sparse.mm(adj_matrix, all_emb[-1])
                all_emb.append(tmp_emb)
            all_emb = torch.stack(all_emb, dim=1)
            mean_emb = torch.mean(all_emb, dim=1)
            user_emb, item_emb = torch.split(mean_emb, [self.num_user, self.num_item])
        else:
            assert False
        return user_emb, item_emb


    def getEmbedding(self, users, pos_items, neg_items):
        # all_users, all_items = self.forward()
        users_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        users_emb_ego = self.user_embeddings(users)
        pos_emb_ego = self.item_embeddings(pos_items)
        neg_emb_ego = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbedding_from_LLM(self, users, pos_items, neg_items):
        # all_users, all_items = self.forward()
        users_emb = self.user_emb_str[users]
        pos_emb = self.item_emb_str[pos_items]
        neg_emb = self.item_emb_str[neg_items]
        users_emb_ego = self.user_str_embeddings(users)
        pos_emb_ego = self.item_str_embeddings(pos_items)
        neg_emb_ego = self.item_str_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego



    def bpr_loss(self, batch, g_type=None):
        users, pos_items, neg_items = batch
        if g_type == 'cf':
            (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        elif g_type == 'llm':
            (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding_from_LLM(users, pos_items, neg_items)
        reg_loss = 1/2 * (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg


    def hsic_graph(self, users, pos_items):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(pos_items)
        input_x = self.user_emb_old[users]
        input_y = self.user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        ### item part ###
        input_i = self.item_emb_old[items]
        input_j = self.item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        loss = loss_user + loss_item
        return loss


    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_infonce_loss(self, embeds1, embeds2, all_embeds2, temp=1.0):
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        # pdb.set_trace()
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss

    def data_to_device(self, batch):
        u, i, j = batch
        u = torch.tensor(u).type(torch.long).to(self.device)  # [batch_size]
        i = torch.tensor(i).type(torch.long).to(self.device)  # [batch_size]
        j = torch.tensor(j).type(torch.long).to(self.device)  # [batch_size]
        batch = [u, i, j]
        return batch

    def calculate_LLaRD_loss(self, batch):
        self.is_training = True
        batch = self.data_to_device(batch)
        # 1. learning denoised u-i graph
        self.masked_adj_matrix = self.graph_learner(self.user_embeddings, self.item_embeddings)
        # 2. learning embeddings from lightgcn
        self.user_emb_old, self.item_emb_old = self.forward(self.adj_matrix, g_type='cf', kr=self.keep_rate)     # original graph
        self.user_emb, self.item_emb = self.forward(self.masked_adj_matrix, g_type='cf', kr=self.keep_rate)         # denoising graph
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(self.user_emb, self.item_emb, batch)
        # 3. prior knowledge from LLM
        # 3.1 prference
        user_prf = self.mlp(self.user_prf)
        item_prf = self.mlp(self.item_prf)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(user_prf, item_prf, batch)
        # 3.2 relation
        self.user_emb_str, self.item_emb_str = self.forward(self.aug_adj_matrix, g_type='llm', kr=self.keep_rate)
        ancstr_embeds, posstr_embeds, negstr_embeds = self._pick_embeds(self.user_emb_str, self.item_emb_str, batch)
        # 4. Max mutual information
        auc, bpr_loss, reg_loss = self.bpr_loss(batch, g_type='cf')
        llm_auc, llm_bpr_loss, llm_reg_loss = self.bpr_loss(batch, g_type='llm')
        prf_loss = self.cal_infonce_loss(anc_embeds, ancprf_embeds, user_prf, self.kd_temperature) + \
            self.cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
            self.cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        prf_loss /= anc_embeds.shape[0]
        prf_loss *= self.prf_weight
        str_cl_loss = self.cal_infonce_loss(anc_embeds, ancstr_embeds, self.user_emb_str, self.kd_temperature) + \
            self.cal_infonce_loss(pos_embeds, posstr_embeds, posstr_embeds, self.kd_temperature) + \
            self.cal_infonce_loss(neg_embeds, negstr_embeds, negstr_embeds, self.kd_temperature)
        str_bpr_loss = (llm_bpr_loss + llm_reg_loss)
        str_cl_loss /= anc_embeds.shape[0]
        str_loss = str_bpr_loss + str_cl_loss*self.delta
        str_loss *= self.str_weight

        # 5. Min mutual information
        users, pos_items, neg_items = batch
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        loss_cf = bpr_loss + reg_loss
        loss_llm = (prf_loss + str_loss)* self.alpha
        loss = loss_cf + loss_llm + ib_loss
        return auc, llm_auc, [bpr_loss, reg_loss], [prf_loss, str_loss], ib_loss, loss
