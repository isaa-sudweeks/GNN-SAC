from copy import deepcopy 

import torch 
import torch.nn as nn 
import torch_geometric 
from torch_geometric.nn import Linear,GCNConv,SAGEConv,GATv2Conv
from torch_geometric.nn import MessagePassing, global_mean_pool
from common.mlp_layers import NormedLinear, mlp 


class GNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: List[int], dropout: float = 0.0):
        super().__init__(aggr = "add") # TODO: make this a config parameter?
        self.phi = mlp(in_channels*2, hidden_channels, out_channels, dropout=dropout)
        self.gamma = mlp(in_channels + out_channels, hidden_channels, out_channels, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i, x_j):
        x_cat = torch.cat([x_i, x_j], dim=1)
        return self.phi(x_cat)
    
    def update(self, aggr_out, x):
        x_cat = torch.cat([x, aggr_out], dim=1)
        return self.gamma(x_cat)

    def __repr__(self):
        repr_str =f"GNN(in_channels={self.in_channels}, out_channels={self.out_channels}, hidden_channels={self.hidden_channels}, dropout={self.dropout})\n"
        repr_str += f"Phi(Message Passing):\t{self.phi}\n"
        repr_str += f"Gamma(Update):\t{self.gamma}\n"
        return repr_str

class Q_GNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: List[int], head_hidden_dims: List[int], dropout: float = 0.0):
        super().__init__(aggr = "add") # TODO: make this a config parameter?
        self.phi = mlp(in_channels*2, hidden_channels, out_channels, dropout=dropout)
        self.gamma = mlp(in_channels + out_channels, hidden_channels, out_channels, dropout=dropout)
        self.head = mlp(out_channels, head_hidden_dims,1)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        graph_embedding = global_mean_pool(x)
        out = self.head(graph_embedding).squeeze(-1)
        return out

    def message(self, x_i, x_j):
        x_cat = torch.cat([x_i, x_j], dim=1)
        return self.phi(x_cat)
    
    def update(self, aggr_out, x):
        x_cat = torch.cat([x, aggr_out], dim=1)
        return self.gamma(x_cat)
    def __repr__(self):
        repr_str =f"Q_GNN(in_channels={self.in_channels}, out_channels={self.out_channels}, hidden_channels={self.hidden_channels}, dropout={self.dropout})\n"
        repr_str += f"Phi(Message Passing):\t{self.phi}\n"
        repr_str += f"Gamma(Update):\t{self.gamma}\n"
        repr_str += f"Pooling:\tglobal_mean_pool\n"
        repr_str += f"Head:\t{self.head}\n"
        return repr_str