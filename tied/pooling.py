from typing import Optional

import torch
from torch import nn


class FirstTokenPooling1D(nn.Module):
    """Takes the first token's embedding."""
    def __init__(self, n_pooling_tokens):
        super().__init__()
        self.n_pooling_tokens = n_pooling_tokens

    def forward(self, x: torch.Tensor):
        return x[:, 0, :]

class NTokenPooling1D(nn.Module):
    """Takes the first n tokens' embeddings."""
    def __init__(self, n_pooling_tokens):
        super().__init__()
        self.n_pooling_tokens = n_pooling_tokens

    def forward(self, x: torch.Tensor):
        return x[:, :self.n_pooling_tokens+1, :]


POOLING2OBJECT = {
    'first': FirstTokenPooling1D,
    'n_token': NTokenPooling1D,
}