from re import X
import torch
import torch.nn as nn
from layer import GraphConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_features) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = nn.Sequential(
            GraphConv(self.input_dim, self.hidden_dim, num_features, is_sparse_inputs=True),
            GraphConv(self.hidden_dim, self.output_dim),
        )

    def forward(self, inputs):
        x, support = inputs
        x = self.layer((x, support))

        return x 



