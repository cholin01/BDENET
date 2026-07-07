import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re, os
from itertools import permutations
from scipy.spatial import distance_matrix
from torch_geometric.data import Data, Batch
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
import ast
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C', 'N', 'O', 'S', 'F', 'P', 'Cl',
            'Br', 'I', 'B', 'Si', 'Fe', 'Zn',
            'Cu', 'Mn', 'Mo', 'other'
        ]) + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`

    return np.array(results)


def calc_bond_features(bond, use_chirality=False):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

    return np.array(bond_feats).astype(int)


def load_mol(molpath, explicit_H=False, use_chirality=True):
    # load mol
    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
    elif re.search(r'.sdf$', molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def smiles_to_graph(data, explicit_H=False, use_chirality=True):
    """
    mol: rdkit.Chem.rdchem.Mol
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    # Add nodes
    try:
        smiles, bde_index, targets = data[0], data[1], data[2]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

            # 2. 检查是否有多个不连通片段

        mol = Chem.AddHs(mol)  # 添加氢原子
        AllChem.EmbedMolecule(mol)  # 生成3D坐标

        if AllChem.MMFFHasAllMoleculeParams(mol):
            # 使用MMFF力场优化
            AllChem.MMFFOptimizeMolecule(mol)

        frags = Chem.GetMolFrags(mol, asMols=True)

        if len(frags) > 1:
            print(f"Multiple fragments found in molecule: {smiles}. Skipping this molecule.")
            return None  # 如果有不连通片段，直接返回 None

        num_atoms = mol.GetNumAtoms()

        atom_feats = np.array([calc_atom_features(atom) for atom in mol.GetAtoms()])

        # print(atom_feats)

        # obtain the positions of the atoms
        atomCoords = mol.GetConformer().GetPositions()

        # Add edges
        src_list = []
        dst_list = []
        bond_feats_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_feats = calc_bond_features(bond)
            # print(f"bond_feats: {bond_feats}")
            bond_lengths = np.array([np.linalg.norm(atomCoords[u] - atomCoords[v])]).astype(float)
            # print(f"bond_len: {bond_lengths}")
            bond_feats = np.concatenate([bond_lengths, bond_feats])
            src_list.extend([u, v])
            dst_list.extend([v, u])
            bond_feats_all.append(bond_feats)
            bond_feats_all.append(bond_feats)

        edge_size = len(src_list)

        g = Data(x=th.tensor(atom_feats, dtype=th.float),
                 edge_index=th.tensor([src_list, dst_list]),
                 pos=th.tensor(atomCoords, dtype=th.float),
                 edge_size=edge_size,
                 bde_index=bde_index, targets=targets,
                 edge_attr=th.tensor(np.array(bond_feats_all), dtype=th.float))

        return g

    except:

        return None


def graph2graph(mol_data):
    x = th.tensor(mol_data['atom_num'], dtype=th.float).unsqueeze(-1)
    edge_index = th.tensor(mol_data['connectivity']).T
    edge_attr = th.tensor(np.array(mol_data['bond']), dtype=th.float).unsqueeze(-1)
    pos = th.tensor(np.array(mol_data['pos']), dtype=th.float)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return g


def load_data(file_path):
    # 读取 CSV 文件
    data_list = []
    df = pd.read_csv(file_path)

    # 转换数据类型
    # df['bde'] = pd.to_numeric(df['bde'], errors='coerce')  # 将 'bde' 列转换为数值型
    # df['bond_index'] = pd.to_numeric(df['bond_index'], errors='coerce')  # 将 'bond_index' 列转换为数值型

    # 提取 SMILES 和目标变量
    smiles = df['molecule'].tolist()
    # frag_1 = df['fragment1'].tolist()
    # frag_2 = df['fragment2'].tolist()
    targets = df['bde'].tolist()
    bde_index = df['bond_index'].tolist()

    for idx in range(len(smiles)):
        data_list.append([smiles[idx], bde_index[idx], targets[idx]])

    return np.array(data_list)


class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        mol_data = self.data[idx]

        # mol_data = smiles_to_graph(self.data[idx])

        # 检查 mol_data 是否为 None 或 NaN
        # if mol_data is None or (isinstance(mol_data, float) and np.isnan(mol_data)):
        #     # 跳过当前项，继续查找下一个
        #     return self.__getitem__(idx - 1)

        # 正常返回 mol_data
        return mol_data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data)

    def train_and_test_split(self, valfrac=0.1, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.labels))
        val_inds = np.random.choice(np.arange(len(self.labels)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.labels)), val_inds)
        return train_inds, val_inds


class BDE2Dataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):

        mol_data = self.data[idx]

        return mol_data

    def collate_fn(self, batch):

        ideal = 0
        bond_index = []

        for idx, data in enumerate(batch):
            edge_size = data.edge_size
            bde_idx = th.tensor(data.bde_index, dtype=th.long)
            bde_idx = bde_idx + ideal
            bond_index.extend(bde_idx)
            ideal = ideal + edge_size

        batch = Batch.from_data_list(batch)

        batch.bde_index = th.tensor(bond_index, dtype=th.long)

        return batch

    def __len__(self):
        return len(self.data)

    def train_and_test_split(self, valfrac=0.1, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.labels))
        val_inds = np.random.choice(np.arange(len(self.labels)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.labels)), val_inds)
        return train_inds, val_inds


class ALGraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data, label):
        self.mol_data = data
        self.y_data = label

    def __getitem__(self, idx):

        mol_graph = graph2graph(self.mol_data[idx])
        y_data = th.tensor(self.y_data[idx])

        return [mol_graph, y_data]

    def collate_fn(self, batch):

        mol_data = []
        labels = []

        for i in range(len(batch)):
            mol_data.append(batch[i][0])
            labels.extend(batch[i][1])

        batch_mol = Batch.from_data_list(mol_data)
        batch_y = th.tensor(labels)

        return batch_mol, batch_y

    def __len__(self):
        return len(self.mol_data)

    def train_and_test_split(self, valfrac=0.1, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.labels))
        val_inds = np.random.choice(np.arange(len(self.labels)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.labels)), val_inds)
        return train_inds, val_inds


def main():
    file_path = '/home/suqun/data/BDE_prediction/filter_error_data.csv'
    data_list = load_data(file_path)
    print(len(data_list))
    # results = []

    results = Parallel(n_jobs=-1)(delayed(smiles_to_graph)(data) for data in tqdm(data_list))

    # for idx, data in tqdm(enumerate(data_list)):
    #
    #     try:
    #         results.append(split_smiles_to_graph(data))
    #     except:
    #         continue

    # results = list(filter(lambda x: x[0] != None, results))
    # print(len(results))
    results = list(filter(lambda x: x != None, results))
    # print(len(results))
    smiles_data = results
    # print(targets, bde_index)
    # np.save("/home/suqun/data/BDE_prediction/data/unique_mol/BDE_labels", (bde_index, targets))
    th.save(smiles_data, "/home/suqun/data/BDE_prediction/data/unique_mol/BDE_smiles_data.pt")


if __name__ == "__main__":
    main()
