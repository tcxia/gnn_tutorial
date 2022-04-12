from audioop import bias
import torch
import torch.nn as nn

import math

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'
