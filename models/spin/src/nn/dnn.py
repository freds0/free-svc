from typing import List

import torch
from torch import nn


class DNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dims: List[int],
        dropout: float = 0.0,
        activation: str = "ReLU",
        activate_last: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = hid_dims[-1]
        self.activate_last = activate_last

        assert len(hid_dims) > 0, len(hid_dims)
        hid_dims = [in_dim] + hid_dims

        self.layers = nn.ModuleList(
            [nn.Linear(hid_dims[i], hid_dims[i + 1]) for i in range(len(hid_dims) - 1)]
        )
        self.num_layer = len(self.layers)
        self.dropout = nn.Dropout(dropout)
        n_acts = self.num_layer - (0 if self.activate_last else 1)
        self.acts = nn.ModuleList([getattr(nn, activation)() for _ in range(n_acts)])

    def forward(self, x: torch.Tensor, x_len: torch.LongTensor = None) -> torch.Tensor:
        for i in range(self.num_layer):
            x = self.layers[i](x)
            if i < self.num_layer - 1 or self.activate_last:
                x = self.dropout(x)
                x = self.acts[i](x)
        return x
