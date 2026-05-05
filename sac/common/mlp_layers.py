from copy import deepcopy

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    """Small critic ensemble used by SAC."""

    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def __len__(self):
        return len(self.modules_list)

    def forward(self, *args, **kwargs):
        return torch.stack([module(*args, **kwargs) for module in self.modules_list], dim=0)

    def copy(self):
        return deepcopy(self)


class NormedLinear(nn.Linear):
    """Linear layer followed by layer norm and activation."""

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = nn.Mish(inplace=False) if act is None else act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """Standard MLP with layer-norm Mish hidden layers."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    final_dropout = dropout if len(dims) == 2 else 0.0
    if act is None:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    else:
        layers.append(NormedLinear(dims[-2], dims[-1], dropout=final_dropout, act=act))
    return nn.Sequential(*layers)
