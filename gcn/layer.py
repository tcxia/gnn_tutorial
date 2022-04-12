import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_features, dropout=0., is_sparse_inputs=False, bias=False, activation=F.relu, featureless=False) -> None:
        super().__init__()

        self.dropout =dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        