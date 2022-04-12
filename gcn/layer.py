import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_features, dropout=0., is_sparse_inputs=False, bias=False, activation=F.relu, featureless=False) -> None:
        super().__init__()

        self.dropout =dropout
        # self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features = num_features

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

        
    def forward(self, inputs):
        x, support = inputs
        
        if self.training and self.is_sparse_inputs:
            x = self.sparse_dropout(x, self.dropout, self.num_features)
        elif self.training:
            x = F.dropout(x, self.dropout)

        if not self.featureless:
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = self.sparse.mm(support, xw)
        
        if self.bias is not None:
            out += self.bias
        
        return self.activation(out), support


    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).byte()
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        out = out * (1. / (1 - rate))

        return out