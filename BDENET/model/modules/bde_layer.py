import torch
import torch.nn as nn
from torch_scatter import scatter_add

"""
FCN for energy prediction
"""

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

class BDELayer(nn.Module):
    def __init__(self, num_in, num_out, activation, zero_init=False):
        super(BDELayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.zero_init = zero_init
        self.linear_diagonal = nn.Linear(self.num_in, self.num_out)
        self.linear_offdiagonal = nn.Linear(self.num_in, self.num_out)
        self.linear_out = nn.Linear(4 * self.num_out + 1, 1)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            nn.init.zeros_(self.linear_diagonal.weight)
            nn.init.zeros_(self.linear_offdiagonal.weight)
            nn.init.zeros_(self.linear_out.weight)
        else:
            nn.init.orthogonal_(self.linear_diagonal.weight)
            nn.init.orthogonal_(self.linear_offdiagonal.weight)
            nn.init.orthogonal_(self.linear_out.weight)

        nn.init.zeros_(self.linear_diagonal.bias)
        nn.init.zeros_(self.linear_offdiagonal.bias)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, fii, fij, edge_index, idx_i, R):
        fii_ = self.activation(self.linear_diagonal(fii[0].squeeze(dim=-1))).squeeze(dim=2).squeeze(dim=0)

        fij_ = self.activation(self.linear_offdiagonal(fij[0].squeeze(dim=-1))).squeeze(dim=2).squeeze(dim=0)

        Nij = unsorted_segment_sum(fij_, idx_i, fii_.size()[0])

        N_t = torch.cat([fii_, Nij], dim=1)

        dist = torch.norm(R[edge_index[0]] - R[edge_index[1]], dim=-1).unsqueeze(-1)

        full_features = torch.concat([N_t[edge_index[0]], dist, N_t[edge_index[1]]], dim=1)

        BDE = self.linear_out(full_features)

        return BDE, full_features
