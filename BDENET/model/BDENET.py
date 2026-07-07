import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3
import torch.nn.functional as F
import logging
from torch_geometric.nn import radius_graph, radius, GINEConv
from .modules.zero_message import ShiftedSoftplus
from .modules.blocks import (
    LinearNodeEmbeddingBlock,
    NonLinearReadoutBlock,
    energy_layer,
    AgnosticNonlinearInteractionBlock,
    RadialEmbeddingBlock,
)

class BDENET(nn.Module):
    """2D+3D Joint Neural network for computing BDE/BDFE (Base + Extra Strategy)"""
    def __init__(
            self,
            # --- 3D Params ---
            order=None,
            basis_functions=None,
            num_basis_functions=128,
            cutoff=None,
            num_elements=1,
            radial_MLP=None,
            avg_num_neighbors=3,
            correlation=3,
            num_interactions=3,
            heads=["dft"],
            hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"),
            MLP_irreps=o3.Irreps("64x0e"),
            gate=ShiftedSoftplus(),
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            
            # --- 2D Params ---
            node_dim_2d=1,      # 2D节点特征维度
            edge_dim_2d=4,      # 2D边特征维度
            hidden_dim_2d=128,  # 2D隐藏层维度
            num_2d_layers=4     # 增加 GIN 的层数
    ):
        super(BDENET, self).__init__()
        self.create_graph = True 
        self.order = order
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
            
        # ==========================================
        # 1. 3D 流初始化 (Equivariant Branch -> Extra)
        # ==========================================
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
        edge_feats_irreps = o3.Irreps('128x0e')
        sh_irreps = o3.Irreps.spherical_harmonics(order)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
            
        self.interactions = torch.nn.ModuleList([
            AgnosticNonlinearInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps if i == 0 else hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            ) for i in range(num_interactions)
        ])
        
        # 移除了 HIL 层
        self.edge_embedding_2d = nn.Embedding(100, 128)
        self.node_embedding_2d = nn.Embedding(100, 128)
        self.edge_embedding = nn.Embedding(100, 8)
        self.mix_embedding = nn.Sequential(nn.Linear(16, 128), nn.Sigmoid())
        self.vec2sca = energy_layer(hidden_irreps, o3.Irreps("128x0e"), gate)
        self.sca_embedding = nn.Sequential(nn.Linear(128, 256))

        # 3D 预测头部 (输出 BDE_extra, BDFE_extra)
        # 节点标量维度是 256，拼接两个节点构成边特征，维度 512
        self.readout_3d = nn.Sequential(
            nn.Linear(512, 256),
            ShiftedSoftplus(),
            nn.Linear(256, 128),
            ShiftedSoftplus(),
            nn.Linear(128, 2)
        )

        # ==========================================
        # 2. 2D 流初始化 (Topological Branch -> Base)
        # ==========================================
        self.node_embed_2d = nn.Embedding(100, hidden_dim_2d)
        self.edge_embed_2d = nn.Linear(edge_dim_2d, hidden_dim_2d)
        
        # 使用 ModuleList 堆叠多层 GINEConv
        self.convs_2d = nn.ModuleList()
        for _ in range(num_2d_layers):
            nn_seq = nn.Sequential(
                nn.Linear(hidden_dim_2d, hidden_dim_2d), 
                nn.ReLU(), 
                nn.Linear(hidden_dim_2d, hidden_dim_2d)
            )
            self.convs_2d.append(GINEConv(nn_seq, edge_dim=hidden_dim_2d))

        # 2D 预测头部 (输出 BDE_base, BDFE_base)
        # 2D 节点维度是 128，拼接两个节点构成边特征，维度 256
        self.readout_2d = nn.Sequential(
            nn.Linear(hidden_dim_2d * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.num_basis_functions = num_basis_functions

    @staticmethod
    def calculate_distances_and_directions(R, idx_i, idx_j):
        Ri = torch.gather(R, -2, idx_i.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)))
        Rj = torch.gather(R, -2, idx_j.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)))
        rij = Rj - Ri 
        dij = torch.norm(rij, dim=-1, keepdim=True)  
        uij = rij / dij  
        return dij, uij

    def forward(self, atoms_batch):
        # 提取 3D 特征
        R = atoms_batch.pos
        Z = atoms_batch.z.unsqueeze(-1)
        Z_2d = atoms_batch.z.long()
        edge_index = atoms_batch.edge_index
        edge_feats = atoms_batch.edge_feats.long()
        
        edge_feats_2d = self.edge_embedding_2d(edge_feats)
        node_feats_2d = self.node_embedding_2d(Z_2d)
        
        for conv in self.convs_2d:
            node_feats_2d = conv(node_feats_2d, edge_index, edge_feats_2d)
            node_feats_2d = F.relu(node_feats_2d)
            
        # 提取边的两个端点特征并拼接
        edge_repr_2d = torch.cat([node_feats_2d[edge_index[0]], node_feats_2d[edge_index[1]]], dim=-1)
        
        # 得到 2D 基础预测值 [num_edges, 2]
        base_preds = self.readout_2d(edge_repr_2d)

        # ==========================================
        # 3D 分支前向传播 (计算 Extra)
        # ==========================================
        edge_feats = self.edge_embedding(edge_feats)
        node_feats = self.node_embedding(Z)
        
        dij, uij = self.calculate_distances_and_directions(R, edge_index[0], edge_index[1])
        rbf = self.radial_embedding(dij)
        
        edge_feats = torch.cat([rbf, edge_feats], dim=-1)
        edge_feats = self.mix_embedding(edge_feats)
        sph = self.spherical_harmonics(uij)
        
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
        node_scalar_3d = self.sca_embedding(node_vc)
        
        # 提取边的两个端点标量特征并拼接
        edge_repr_3d = torch.cat([node_scalar_3d[edge_index[0]], node_scalar_3d[edge_index[1]]], dim=-1)
        
        # 得到 3D 额外修正值 [num_edges, 2]
        extra_preds = self.readout_3d(edge_repr_3d)

        # ==========================================
        # 最终预测：Base + Extra
        # ==========================================
        final_preds = base_preds + extra_preds
        
        return final_preds[:, 0].squeeze(-1), final_preds[:, 1].squeeze(-1)
