###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from typing import Dict, Optional, Union
from .symmetric_contraction import SymmetricContraction
from .irreps_tools import *
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch
from math import pi as PI
from .zero_message import *

from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)


class HIL(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int):
        super(HIL, self).__init__()

        self.in_channels = 256
        self.out_channels = 256
        self.cutoff = 5.0
        self.num_layers = 8

        # self.mlp_radial = torch.nn.Sequential(torch.nn.Linear(8, self.in_channels), torch.nn.SiLU())

        self.aggregate = torch.nn.ModuleList([node_update(256, 256) for _ in range(self.num_layers)])
        self.message = torch.nn.ModuleList([edge_update(256, 256, 128, 5.0) for _ in range(self.num_layers)])

    def forward(self, node_feats, edge_feats, edge_index, dist):
        for mess, agg in zip(self.message, self.aggregate):
            edge_attr = mess(node_feats, dist, edge_feats, edge_index)
            node_feats = agg(node_feats, edge_attr, edge_index)

        return node_feats, edge_attr


def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
        reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irrep_out)

    def forward(
            self,
            x: torch.Tensor,
            heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[Callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
            num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irrep_out)

    def forward(
            self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


class HiddenReadoutBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[Callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
            num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])

    def forward(
            self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return x  # [n_nodes, len(heads)]


class energy_layer(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[Callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
            num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        # self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irrep_out)

    def forward(
            self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return x


class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
            self,
            node_feats_irreps: o3.Irreps,
            target_irreps: o3.Irreps,
            correlation: int,
            use_sc: bool = True,
            num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
            self,
            node_feats: torch.Tensor,
            sc: Optional[torch.Tensor],
            node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            node_attrs_irreps: o3.Irreps,
            node_feats_irreps: o3.Irreps,
            edge_attrs_irreps: o3.Irreps,
            edge_feats_irreps: o3.Irreps,
            target_irreps: o3.Irreps,
            hidden_irreps: o3.Irreps,
            avg_num_neighbors: float,
            radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            idx: int,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
            radial_type: str = "bessel",
            distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
            # node_attrs: torch.Tensor,
            # edge_index: torch.Tensor,
            # atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        # if hasattr(self, "distance_transform"):
        #     edge_lengths = self.distance_transform(
        #         edge_lengths, node_attrs, edge_index, atomic_numbers
        #     )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
            self,
            sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
            edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


class Spherical_block(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, gate=torch.nn.functional.silu):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.pre_activation = nn.Activation(irreps_in=self.irreps_in, acts=[gate])
        self.pre_linear = o3.Linear(
            self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True
        )
        self.post_activation = nn.Activation(irreps_in=self.irreps_in, acts=[gate])
        self.post_linear = o3.Linear(
            self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True
        )

    def forward(self, xs):
        ys = xs
        xs = self.pre_activation(xs)
        xs = self.pre_linear(xs)
        xs = self.post_activation(xs)
        xs = self.post_linear(xs)
        xs = ys + xs

        return xs


class Pair_block(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.node_feats_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.node_feats_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(self, node_feats_i: torch.Tensor, node_feats_j: torch.Tensor, edge_feats: torch.Tensor,
                edge_index: torch.Tensor, num_nodes) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats_i, node_feats_j, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=sender, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message)
        # print(message.size())
        return message


class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.value = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        # input_dim = 128
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(self,
                idx: int,
                node_attrs: torch.Tensor,
                node_feats: torch.Tensor,
                edge_attrs: torch.Tensor,
                edge_feats: torch.Tensor,
                edge_index: torch.Tensor,
                ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        node_feats = self.value(node_feats)

        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        # if idx > 0:
        #     message = message + node_feats
        return message  # [n_nodes, irreps]


class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
            self,
            idx: int,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        # print(edge_feats.dtype)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        # print(message.size())
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
                torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"
