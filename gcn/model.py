import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout) -> None:
        super().__init__()

        self.gc1 = GraphConv(input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



