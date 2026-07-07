from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from typing import List, Optional
import logging


class edge_update(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(edge_update, self).__init__()
        self.cutoff = cutoff
        self.linear = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, node_feats, dist, rbf, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(rbf) * C.view(-1, 1)
        node_feats = self.linear(node_feats)
        edge_feats = node_feats[j] * W
        return edge_feats


class node_update(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(node_update, self).__init__()
        self.act = ShiftedSoftplus()
        self.linear_1 = Linear(num_filters, hidden_channels)
        self.linear_2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_2.bias.data.fill_(0)

    def forward(self, node_feats, edge_feats, edge_index):
        _, i = edge_index
        agg = scatter(edge_feats, i, dim=0, dim_size=node_feats.size()[0])
        agg = self.linear_1(agg)
        agg = self.act(agg)
        agg = self.linear_2(agg)

        return node_feats + agg


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift