import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3
import torch.nn.functional as F
import logging
from torch_geometric.nn import radius_graph, radius
from torch_geometric.nn import radius_graph
from .modules.zero_message import ShiftedSoftplus

from .modules.blocks import (
    LinearNodeEmbeddingBlock,
    NonLinearReadoutBlock,
    energy_layer,
    AgnosticNonlinearInteractionBlock,
    RadialEmbeddingBlock,
    HIL,
)



class DecNet(nn.Module):
    """Neural network for computing Hamiltonian/Overlap matrices in a rotationally equivariant way"""

    def __init__(
            self,
            order=None,  # 1 maximum order of spherical harmonics features
            basis_functions=None,  # exp-bernstein
            num_basis_functions=128,
            # type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            cutoff=None,  # 15.0 cutoff distance (default is 15 Bohr)
            num_elements=1,
            radial_MLP=None,
            avg_num_neighbors=3,
            correlation=3,
            num_interactions=3,
            heads=["dft"],
            # hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e+128x3o+128x4e"),
            hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"),
            MLP_irreps=o3.Irreps("64x0e"),
            gate=ShiftedSoftplus(),
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
    ):
        super(DecNet, self).__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian/core_hamiltonian/overlap_matrix/energy/forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        self.order = order

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        # edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        edge_feats_irreps = o3.Irreps('128x0e')

        sh_irreps = o3.Irreps.spherical_harmonics(order)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        inter = AgnosticNonlinearInteractionBlock(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps
            inter = AgnosticNonlinearInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)

        # self.scalar_interactions = torch.nn.ModuleList()
        #
        # for i in range(2):
        #     scalar_inter = HIL(128, 128)
        #     self.scalar_interactions.append(scalar_inter)
        self.scalar_inter = HIL(128, 128)

        self.edge_embedding = nn.Linear(1, 8)
        self.mix_embedding = nn.Sequential(nn.Linear(16, 128), nn.Sigmoid())

        self.vec2sca = energy_layer(hidden_irreps, o3.Irreps("128x0e"), gate)

        self.sca_embedding = nn.Sequential(nn.Linear(128, 256))

        self.readout_layer = nn.Sequential(
            # nn.Linear(1024, 512),
            # ShiftedSoftplus(),
            nn.Linear(512, 256),
            ShiftedSoftplus(),
            nn.Linear(256, 128),
            ShiftedSoftplus(),
            nn.Linear(128, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 1)
            )

        self.energy_output = NonLinearReadoutBlock(
            hidden_irreps,
            (len(heads) * MLP_irreps).simplify(),
            gate,
            o3.Irreps(f"{len(heads)}x0e"),
            len(heads),
        )

        # store hyperparameter values
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.num_basis_functions = num_basis_functions

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    @staticmethod
    def calculate_distances_and_directions(R, idx_i, idx_j):
        Ri = torch.gather(
            R,
            -2,
            idx_i.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )

        Rj = torch.gather(
            R,
            -2,
            idx_j.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )

        # Ri = R[idx_i]
        # Rj = R[idx_j]
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return dij, uij

    """
    Computes the Hamiltonian/Overlap matrix

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        matrix: Hamiltonian/Overlap matrix of shape [batch_size, num_orbitals, num_orbitals]
    """  # coding: utf-8

    def forward(self, atoms_batch, mean, std):

        R = atoms_batch.pos
        Z = atoms_batch.z.unsqueeze(-1)
        batch = atoms_batch.batch
        edge_feats = atoms_batch.edge_feats
        edge_index = atoms_batch.edge_index

        # edge_index, edge_feats = _extend_to_radius_graph(R, local_edge_index, edge_feats, 3.0, batch, unspecified_type_number=0, is_sidechain=None)
        edge_feats = edge_feats.unsqueeze(-1)

        edge_feats = edge_feats.float()

        edge_feats = self.edge_embedding(edge_feats)

        node_feats = self.node_embedding(Z)

        # compute radial basis functions and spherical harmonics
        dij, uij = self.calculate_distances_and_directions(R, edge_index[0], edge_index[1])
        rbf = self.radial_embedding(dij)

        edge_feats = torch.cat([rbf, edge_feats], dim=-1)

        edge_feats = self.mix_embedding(edge_feats)

        sph = self.spherical_harmonics(uij)

        # Interactions
        for idx, interaction in enumerate(self.interactions):
            node_feats = interaction(
                idx=idx,
                node_attrs=Z,
                node_feats=node_feats,
                edge_attrs=sph,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

        node_vc = self.vec2sca(node_feats)

        sca_embedding = self.sca_embedding(node_vc)

        node_scalar, edge_attr = self.scalar_inter(node_feats=sca_embedding, edge_feats=edge_feats, edge_index=edge_index, dist=dij)

        # node_out = torch.cat([sca_embedding, node_scalar], dim=-1)

        edge_feats = torch.cat([node_scalar[edge_index[0]], node_scalar[edge_index[1]]], dim=-1)

        BDE = self.readout_layer(edge_feats)

        BDE = BDE * std + mean

        return BDE.squeeze(-1)




