import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GraphAttentionLayer, SpGraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads) -> None:
        super().__init__()
        self.dropout = dropout

        self.attns = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attn in enumerate(self.attns):
            self.add_module("attn_{}".format(i), attn)
        
        self.out_attn = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attn(x, adj) for attn in self.attns], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attn(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads) -> None:
        super().__init__()
        self.dropout = dropout
        self.attns = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attn in enumerate(self.attns):
            self.add_module("attn_{}".format(i), attn)

        self.out_attn = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attn(x, adj) for attn in self.attns], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attn(x, adj))
        return F.log_softmax(x, dim=1)
        
