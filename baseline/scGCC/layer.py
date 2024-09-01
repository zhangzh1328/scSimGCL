import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# full connected layer
def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )

class GATEncoder(nn.Module):

    def __init__(self, num_genes, latent_dim, num_heads=20
                 , dropout=0.4, fc=None):
        super(GATEncoder, self).__init__()
        # initialize parameter
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.gat_layer_1 = GATConv(
            in_channels=num_genes, out_channels=128,
            heads=num_heads,
            dropout=dropout,
            concat=True)
        in_dim2 = 128 * num_heads

        self.gat_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=latent_dim,
            heads=num_heads,
            concat=False)

        self.fc = fc

    def forward(self, x, edge_index):
        # x = x.float()
        hidden_out1 = self.gat_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out1 = F.dropout(hidden_out1, p=0.4, training=self.training)
        hidden_out2 = self.gat_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        embedding = hidden_out2
        # add project head
        if self.fc is not None:
            embedding = self.fc(embedding)
        return embedding

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, num_genes=3000, dim=256, r=512, m=0.99, T=0.2, head=20, mlp=False):
        """
        dim: feature dimension
        r: queue size
        m: momentum for updating key encoder
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)
        self.encoder_k = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)

        # 1、create mlp
        if mlp:
            self.encoder_q.fc = nn.Sequential(full_block(dim, 512, 0.4), full_block(512, dim))
            self.encoder_k.fc = nn.Sequential(full_block(dim, 512, 0.4), full_block(512, dim))
        # initialize
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, edge_index=None, is_eval=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return embeddings (used for clustering)

        Output:
            logits, targets
        """

        if is_eval:
            k = self.encoder_k(im_q, edge_index)
            k = nn.functional.normalize(k, dim=1)
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k, edge_index)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute query features
        q = self.encoder_q(im_q, edge_index)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # N表示一个batch样本数
        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
