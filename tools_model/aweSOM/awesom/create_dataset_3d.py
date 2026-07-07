import os
import shutil
import re
from ast import literal_eval
from typing import Callable, List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class SOM(InMemoryDataset):
    """PyTorch Geometric Dataset for site-of-metabolism prediction from SD-Files or SMILES files."""

    def __init__(
        self,
        root: str,
        labeled: bool = True,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], Data]] = None,
        # 🌟 核心修改 1：引入 BDE 和 BDFE 的全局均值和标准差
        mean_bde: float = 94.62390701792717,
        std_bde: float = 11.503478622893386,
        mean_bdfe: float = 84.48772566704848,
        std_bdfe: float = 12.50514247866427,
    ) -> None:
        self.labeled = labeled
        self.mean_bde = mean_bde
        self.std_bde = std_bde
        self.mean_bdfe = mean_bdfe
        self.std_bdfe = std_bdfe

        # Delete the processed folder if it exists
        processed_folder = os.path.join(root, "processed")
        if os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)
            print(f"Deleted existing processed folder at: {processed_folder}")

        super().__init__(root, transform, pre_transform, pre_filter)

        self.root = root
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def find_input_file(
        self, extensions: List[str] = [".sdf", ".smi", ".smiles"]
    ) -> Optional[str]:
        """Finds the first file in the root directory with one of the allowed extensions."""
        for file_name in os.listdir(self.root):
            if any(file_name.endswith(ext) for ext in extensions):
                return os.path.join(self.root, file_name)
        return None

    def process(self) -> None:
        input_file = self.find_input_file()
        if input_file is None:
            raise NotImplementedError(
                "Data file must be either .sdf, .smi, or .smiles."
            )

        data_list = self.data_processing(input_file=input_file)
        torch.save((self.collate(data_list)), self.processed_paths[0])

    def data_processing(self, input_file: str) -> List[Data]:
        """Process the input file and create Data objects."""
        _, file_extension = os.path.splitext(input_file)

        molecules, labels, descriptions = self.load_molecules(
            input_file, file_extension
        )

        data_list = []
        for mol_id, (mol, soms, description) in enumerate(
            zip(molecules, labels, descriptions)
        ):
            if mol is None:
                continue

            # 不再去除氫原子，僅分配 SOM 標籤
            mol, soms = self.assign_som_labels(mol, soms)

            if len(soms) == 0 and self.labeled:
                continue  # Skip molecules without SoMs in labeled mode

            data = self.mol_to_data(mol, soms, mol_id, description)
            if data is not None:
                data_list.append(data)

        return data_list

    def load_molecules(
        self, input_file: str, file_extension: str
    ) -> tuple[List[Chem.Mol], List[List[int]], List[str]]:
        """Load molecules from file based on extension."""
        molecules: List[Chem.Mol] = []
        labels: List[List[int]] = []
        descriptions: List[str] = []

        if file_extension == ".sdf":
            # 這裡設定 removeHs=False 保留了從文件中讀取的 H 原子
            suppl = Chem.SDMolSupplier(input_file, removeHs=False)
            for mol in suppl:
                if mol is None:
                    continue

                soms = []
                if self.labeled:
                    soms_prop = mol.GetProp("soms") if mol.HasProp("soms") else "[]"
                    soms = literal_eval(soms_prop)

                desc = (
                    mol.GetProp("_Name")
                    if mol.HasProp("_Name")
                    else f"{len(molecules)}"
                )

                molecules.append(mol)
                labels.append(soms)
                descriptions.append(desc)

        elif file_extension in [".smi", ".smiles"]:
            with open(input_file, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split("\t")
                    smiles = parts[0]

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    soms = []
                    if self.labeled and len(parts) > 2:
                        soms_str = parts[2]
                        try:
                            soms = literal_eval(soms_str)
                        except ValueError:
                            soms = []

                    desc = parts[1] if len(parts) > 1 else f"{line_num}"

                    molecules.append(mol)
                    labels.append(soms)
                    descriptions.append(desc)
        else:
            raise NotImplementedError(f"Invalid file extension: {file_extension}")

        return molecules, labels, descriptions

    def assign_som_labels(
        self, mol: Chem.Mol, soms: List[int]
    ) -> tuple[Chem.Mol, List[int]]:
        """保留氫原子，直接給對應索引的原子打上標籤。"""
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            # 記錄 1-based 索引，用於與 SDF 的 BDE 文本對齊
            atom.SetIntProp("orig_idx", atom_id + 1)
            
            if atom_id in soms:
                atom.SetIntProp("label", 1)
            else:
                atom.SetIntProp("label", 0)

        return mol, soms

    def mol_to_data(
        self, mol: Chem.Mol, soms: List[int], mol_id: int, description: str
    ) -> Optional[Data]:
        """Convert a molecule to a PyTorch Geometric Data object."""
        try:
            # 使用正則解析 SDF 文件中的 BDE 和 BDFE 字符串
            bde_dict = {}
            bdfe_dict = {}

            if mol.HasProp("Prediction_BDE"):
                for line in mol.GetProp("Prediction_BDE").strip().split('\n'):
                    if ':' in line:
                        b_str, v_str = line.split(':')
                        parts = b_str.split('-')
                        if len(parts) == 2:
                            u = int(re.search(r'\d+', parts[0]).group())
                            v = int(re.search(r'\d+', parts[1]).group())
                            val = float(v_str.strip())
                            bde_dict[(u, v)] = val
                            bde_dict[(v, u)] = val

            if mol.HasProp("Prediction_BDFE"):
                for line in mol.GetProp("Prediction_BDFE").strip().split('\n'):
                    if ':' in line:
                        b_str, v_str = line.split(':')
                        parts = b_str.split('-')
                        if len(parts) == 2:
                            u = int(re.search(r'\d+', parts[0]).group())
                            v = int(re.search(r'\d+', parts[1]).group())
                            val = float(v_str.strip())
                            bdfe_dict[(u, v)] = val
                            bdfe_dict[(v, u)] = val


            # 1. 提取節點特徵
            atom_features = []
            atom_ids = []
            som_labels = []

            for atom in mol.GetAtoms():
                atom_id = atom.GetIdx()
                features = self.get_atom_features(atom)
                atom_features.append(features)
                atom_ids.append(atom_id)
                som_labels.append(1 if atom_id in soms else 0)

            # 2. 提取邊特徵與構建無向圖的雙向邊
            edge_index_list = []
            edge_attr_list = []

            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                # 獲取原子的 1-based 索引
                orig_u = bond.GetBeginAtom().GetIntProp("orig_idx")
                orig_v = bond.GetEndAtom().GetIntProp("orig_idx")
                
                # 🌟 核心修改 2：獲取原始數值，若缺失則預設為全局平均值 (Mean)
                raw_bde = bde_dict.get((orig_u, orig_v), self.mean_bde)
                raw_bdfe = bdfe_dict.get((orig_u, orig_v), self.mean_bdfe)

                # 🌟 核心修改 3：Z-Score 標準化
                norm_bde = (raw_bde - self.mean_bde) / self.std_bde
                norm_bdfe = (raw_bdfe - self.mean_bdfe) / self.std_bdfe

                # 將標準化後的值傳入
                bond_features = self.get_bond_features(bond, norm_bde, norm_bdfe)
                
                # 添加正向邊和反向邊
                edge_index_list.append([begin_idx, end_idx])
                edge_index_list.append([end_idx, begin_idx])
                
                edge_attr_list.append(bond_features)
                edge_attr_list.append(bond_features)

            # 3. 轉換為 Tensor
            x = torch.tensor(atom_features, dtype=torch.float32)
            edge_index = (
                torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            )
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            y = torch.tensor(som_labels, dtype=torch.long)
            mol_ids = torch.full((len(atom_ids),), mol_id, dtype=torch.long)
            atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long)

            # 4. 封裝為 PyG Data 物件
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                mol_id=mol_ids,
                atom_id=atom_ids_tensor,
            )
            data.description = description

            return data

        except Exception as e:
            print(f"Error processing molecule {description}: {e}")
            return None

    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """Generate atom features."""
        atomic_num = atom.GetAtomicNum()
        element_list = [
            1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53,
        ]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I

        features = []
        for element in element_list:
            features.append(1.0 if atomic_num == element else 0.0)
        
        features.append(1.0 if atomic_num not in element_list else 0.0)

        return features

    def get_bond_features(self, bond: Chem.Bond, bde_val: float, bdfe_val: float) -> List[float]:
        """Generate bond features."""
        bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        bond_type_str = str(bond.GetBondType())

        features = []
        for bond_type in bond_types:
            features.append(1.0 if bond_type_str == bond_type else 0.0)
        features.append(1.0 if bond_type_str not in bond_types else 0.0)

        features.append(1.0 if bond.IsInRing() else 0.0)
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        
        # 🌟 寫入標準化後的 BDE 和 BDFE 數值
        features.append(bde_val)
        features.append(bdfe_val)

        return features
