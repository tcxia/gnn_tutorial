import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_features) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        