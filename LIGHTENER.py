import torch
import math
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.nn.conv import GCNConv
from utils import device


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, dropout=0):
        super(GraphConstructor, self).__init__()
        assert input_dim % h == 0

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def DataAug(x, adj, prob_feature, prob_edge):
    batch_size = x.shape[0]
    input_dim = x.shape[1]

    tensor_p = torch.ones((batch_size, input_dim)) * (1 - prob_feature)
    mask_feature = torch.bernoulli(tensor_p).to(device)

    tensor_p = torch.ones((batch_size, batch_size)) * (1 - prob_edge)
    mask_edge = torch.bernoulli(tensor_p).to(device)

    return mask_feature * x, mask_edge * adj


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)


def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm)

    return loss


class Model(nn.Module):
    def __init__(self, input_dim, graph_head, phi, gcn_dim, mlp_dim,
                 prob_feature, prob_edge, tau, alpha, beta, dropout):
        super(Model, self).__init__()
        self.prob_feature = prob_feature
        self.prob_edge = prob_edge
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.graphconstructor = GraphConstructor(input_dim, graph_head, phi, dropout=0)
        self.gcn = GCNConv(input_dim, gcn_dim)
        self.w_imp = nn.Linear(gcn_dim, input_dim)
        self.mlp = nn.Linear(gcn_dim, mlp_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        adj = self.graphconstructor(x, x)
        adj = adj - torch.diag_embed(adj.diag())
        edge_index = torch.nonzero(adj == 1).T

        x_aug, adj_aug = DataAug(x, adj, self.prob_feature, self.prob_edge)
        edge_index_aug = torch.nonzero(adj_aug == 1).T

        z = self.gcn(x, edge_index)
        z_aug = self.gcn(x_aug, edge_index_aug)
        x_imp = self.w_imp(z)

        z_mlp = self.mlp(z)
        z_mlp_aug = self.mlp(z_aug)
        loss_cl = final_cl_loss(self.alpha, self.beta, z_mlp, z_mlp_aug, adj, adj_aug, self.tau, hidden_norm=True)

        return z, x_imp, loss_cl

