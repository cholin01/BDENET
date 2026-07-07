import os
import re
import shutil
from ast import literal_eval
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class SOM(InMemoryDataset):
    """
    PyTorch Geometric Dataset for site-of-metabolism prediction.

    This version additionally exports:
        1. atom_feature_table.csv
        2. bond_bde_table.csv

    Important:
        The actual node features and edge features are kept compatible with
        the original model checkpoints.

    Edge feature order:
        bond type one-hot:
            SINGLE, DOUBLE, TRIPLE, AROMATIC, OTHER
        ring flag
        conjugated flag
        normalized BDE
        normalized BDFE
    """

    def __init__(
        self,
        root: str,
        labeled: bool = True,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], Data]] = None,
        mean_bde: float = 94.62390701792717,
        std_bde: float = 11.503478622893386,
        mean_bdfe: float = 84.48772566704848,
        std_bdfe: float = 12.50514247866427,
        bde_hot_quantile: float = 0.20,
        som_index_base: str = "zero",  # zero, one, auto
        force_reprocess: bool = False,
        rebuild_if_metadata_missing: bool = True,
    ) -> None:
        self.labeled = labeled
        self.mean_bde = float(mean_bde)
        self.std_bde = float(std_bde)
        self.mean_bdfe = float(mean_bdfe)
        self.std_bdfe = float(std_bdfe)
        self.bde_hot_quantile = float(bde_hot_quantile)
        self.som_index_base = som_index_base
        self.rebuild_if_metadata_missing = rebuild_if_metadata_missing

        if force_reprocess:
            processed_folder = os.path.join(root, "processed")
            if os.path.exists(processed_folder):
                shutil.rmtree(processed_folder)
                print(f"[SOM] Deleted existing processed folder: {processed_folder}")

        super().__init__(root, transform, pre_transform, pre_filter)

        self._load_processed()

        if (
            rebuild_if_metadata_missing
            and (len(self.atom_records) == 0 or len(self.bond_records) == 0)
        ):
            print(
                "[SOM] Metadata tables are empty. "
                "This usually means an old processed/data.pt was loaded. "
                "Rebuilding processed data..."
            )
            processed_folder = self.processed_dir
            if os.path.exists(processed_folder):
                shutil.rmtree(processed_folder)
            os.makedirs(processed_folder, exist_ok=True)
            self.process()
            self._load_processed()

    def _load_processed(self) -> None:
        loaded = torch.load(self.processed_paths[0], weights_only=False)

        if isinstance(loaded, tuple) and len(loaded) == 3:
            data_slices, atom_records, bond_records = loaded
            self.data, self.slices = data_slices
            self.atom_records = atom_records
            self.bond_records = bond_records
        else:
            self.data, self.slices = loaded
            self.atom_records = []
            self.bond_records = []
            print(
                "[SOM] Warning: loaded an old processed file without "
                "atom_records and bond_records."
            )

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def find_input_file(
        self,
        extensions: List[str] = [".sdf", ".smi", ".smiles"],
    ) -> Optional[str]:
        for file_name in os.listdir(self.root):
            if any(file_name.endswith(ext) for ext in extensions):
                return os.path.join(self.root, file_name)
        return None

    def process(self) -> None:
        print("[SOM] Processing raw input file and rebuilding metadata tables...")

        input_file = self.find_input_file()
        if input_file is None:
            raise FileNotFoundError("Data file must be .sdf, .smi, or .smiles.")

        data_list, atom_records, bond_records = self.data_processing(input_file)

        if len(data_list) == 0:
            raise RuntimeError("No valid molecules were processed.")

        torch.save(
            (self.collate(data_list), atom_records, bond_records),
            self.processed_paths[0],
        )

        print(f"[SOM] Processed molecules: {len(data_list)}")
        print(f"[SOM] Atom metadata rows: {len(atom_records)}")
        print(f"[SOM] Bond metadata rows: {len(bond_records)}")

    def export_metadata_tables(
        self,
        output_dir: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        os.makedirs(output_dir, exist_ok=True)

        atom_df = pd.DataFrame(self.atom_records)
        bond_df = pd.DataFrame(self.bond_records)

        atom_df.to_csv(os.path.join(output_dir, "atom_feature_table.csv"), index=False)
        bond_df.to_csv(os.path.join(output_dir, "bond_bde_table.csv"), index=False)

        return atom_df, bond_df

    def data_processing(
        self,
        input_file: str,
    ) -> Tuple[List[Data], List[Dict[str, Any]], List[Dict[str, Any]]]:
        _, file_extension = os.path.splitext(input_file)

        molecules, labels, descriptions = self.load_molecules(
            input_file=input_file,
            file_extension=file_extension,
        )

        data_list: List[Data] = []
        all_atom_records: List[Dict[str, Any]] = []
        all_bond_records: List[Dict[str, Any]] = []

        for raw_mol_id, (mol, soms_raw, description) in enumerate(
            zip(molecules, labels, descriptions)
        ):
            if mol is None:
                continue

            soms = self.normalize_som_indices(mol, soms_raw)

            if self.labeled and len(soms) == 0:
                continue

            mol_id = len(data_list)

            data, atom_records, bond_records = self.mol_to_data(
                mol=mol,
                soms=soms,
                mol_id=mol_id,
                raw_mol_id=raw_mol_id,
                description=description,
            )

            if data is None:
                continue

            data_list.append(data)
            all_atom_records.extend(atom_records)
            all_bond_records.extend(bond_records)

        return data_list, all_atom_records, all_bond_records

    def load_molecules(
        self,
        input_file: str,
        file_extension: str,
    ) -> Tuple[List[Chem.Mol], List[List[int]], List[str]]:
        molecules: List[Chem.Mol] = []
        labels: List[List[int]] = []
        descriptions: List[str] = []

        if file_extension == ".sdf":
            suppl = Chem.SDMolSupplier(input_file, removeHs=False)

            for i, mol in enumerate(suppl):
                if mol is None:
                    continue

                soms: List[int] = []
                if self.labeled:
                    soms_prop = mol.GetProp("soms") if mol.HasProp("soms") else "[]"
                    soms = self.parse_som_list(soms_prop)

                desc = mol.GetProp("_Name") if mol.HasProp("_Name") else str(i)

                molecules.append(mol)
                labels.append(soms)
                descriptions.append(desc)

        elif file_extension in [".smi", ".smiles"]:
            with open(input_file, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if "\t" in line:
                        parts = line.split("\t")
                    else:
                        parts = line.split(maxsplit=2)

                    smiles = parts[0]
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is None:
                        print(f"[SOM] Skip invalid SMILES at line {line_num}: {smiles}")
                        continue

                    desc = parts[1] if len(parts) > 1 else str(line_num)

                    soms: List[int] = []
                    if self.labeled and len(parts) > 2:
                        soms = self.parse_som_list(parts[2])

                    molecules.append(mol)
                    labels.append(soms)
                    descriptions.append(desc)

        else:
            raise NotImplementedError(f"Invalid file extension: {file_extension}")

        return molecules, labels, descriptions

    @staticmethod
    def parse_som_list(text: Any) -> List[int]:
        if text is None:
            return []

        if isinstance(text, (list, tuple)):
            try:
                return [int(x) for x in text]
            except Exception:
                return []

        text = str(text).strip()
        if text == "":
            return []

        try:
            obj = literal_eval(text)
            if isinstance(obj, int):
                return [int(obj)]
            if isinstance(obj, (list, tuple, set)):
                return [int(x) for x in obj]
        except Exception:
            pass

        nums = re.findall(r"-?\d+", text)
        return [int(x) for x in nums]

    def normalize_som_indices(self, mol: Chem.Mol, soms_raw: List[int]) -> List[int]:
        if not self.labeled:
            return []

        n_atoms = mol.GetNumAtoms()

        try:
            soms = [int(x) for x in soms_raw]
        except Exception:
            return []

        if self.som_index_base == "one":
            soms = [x - 1 for x in soms]

        elif self.som_index_base == "auto":
            if len(soms) > 0:
                if 0 not in soms and min(soms) >= 1 and max(soms) <= n_atoms:
                    soms = [x - 1 for x in soms]

        elif self.som_index_base == "zero":
            pass

        else:
            raise ValueError(
                f"Invalid som_index_base={self.som_index_base}. "
                "Use zero, one, or auto."
            )

        soms = sorted(set([x for x in soms if 0 <= x < n_atoms]))
        return soms

    @staticmethod
    def safe_smiles(mol: Chem.Mol) -> str:
        try:
            mol_no_h = Chem.RemoveHs(Chem.Mol(mol))
            return Chem.MolToSmiles(mol_no_h, canonical=True)
        except Exception:
            return Chem.MolToSmiles(mol, canonical=True)

    @staticmethod
    def parse_prediction_property(
        mol: Chem.Mol,
        prop_name: str,
    ) -> Dict[Tuple[int, int], float]:
        """
        Parse SDF property like:
            C1-H2: 96.2
            1-2: 96.2

        Atom indices in the text are treated as 1-based.
        """

        out: Dict[Tuple[int, int], float] = {}

        if not mol.HasProp(prop_name):
            return out

        text = mol.GetProp(prop_name).strip()
        if not text:
            return out

        for line in text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            left, right = line.split(":", 1)
            ids = re.findall(r"\d+", left)

            if len(ids) < 2:
                continue

            try:
                u = int(ids[0])
                v = int(ids[1])
                val = float(right.strip())
            except Exception:
                continue

            out[(u, v)] = val
            out[(v, u)] = val

        return out

    def mol_to_data(
        self,
        mol: Chem.Mol,
        soms: List[int],
        mol_id: int,
        raw_mol_id: int,
        description: str,
    ) -> Tuple[Optional[Data], List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            smiles = self.safe_smiles(mol)

            for atom in mol.GetAtoms():
                atom.SetIntProp("orig_idx", atom.GetIdx() + 1)
                atom.SetIntProp("label", 1 if atom.GetIdx() in soms else 0)

            bde_dict = self.parse_prediction_property(mol, "Prediction_BDE")
            bdfe_dict = self.parse_prediction_property(mol, "Prediction_BDFE")

            atom_features: List[List[float]] = []
            atom_ids: List[int] = []
            som_labels: List[int] = []

            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                atom_features.append(self.get_atom_features(atom))
                atom_ids.append(atom_idx)
                som_labels.append(1 if atom_idx in soms else 0)

            edge_index_list: List[List[int]] = []
            edge_attr_list: List[List[float]] = []
            bond_records: List[Dict[str, Any]] = []

            for bond in mol.GetBonds():
                bond_idx = bond.GetIdx()

                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()

                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                orig_u = begin_atom.GetIntProp("orig_idx")
                orig_v = end_atom.GetIntProp("orig_idx")

                bde_missing = (orig_u, orig_v) not in bde_dict
                bdfe_missing = (orig_u, orig_v) not in bdfe_dict

                raw_bde = bde_dict.get((orig_u, orig_v), self.mean_bde)
                raw_bdfe = bdfe_dict.get((orig_u, orig_v), self.mean_bdfe)

                norm_bde = (raw_bde - self.mean_bde) / self.std_bde
                norm_bdfe = (raw_bdfe - self.mean_bdfe) / self.std_bdfe

                bond_features = self.get_bond_features(
                    bond=bond,
                    bde_val=norm_bde,
                    bdfe_val=norm_bdfe,
                )

                # Keep original behavior: explicitly add both directions.
                edge_index_list.append([begin_idx, end_idx])
                edge_index_list.append([end_idx, begin_idx])

                edge_attr_list.append(bond_features)
                edge_attr_list.append(bond_features)

                bond_records.append(
                    {
                        "mol_id": mol_id,
                        "raw_mol_id": raw_mol_id,
                        "description": description,
                        "smiles": smiles,
                        "bond_idx": bond_idx,
                        "begin_atom_idx": begin_idx,
                        "end_atom_idx": end_idx,
                        "begin_orig_idx_1based": orig_u,
                        "end_orig_idx_1based": orig_v,
                        "begin_atom_symbol": begin_atom.GetSymbol(),
                        "end_atom_symbol": end_atom.GetSymbol(),
                        "bond_type": str(bond.GetBondType()),
                        "is_ring": int(bond.IsInRing()),
                        "is_aromatic": int(bond.GetIsAromatic()),
                        "is_conjugated": int(bond.GetIsConjugated()),
                        "bde_value": float(raw_bde),
                        "bdfe_value": float(raw_bdfe),
                        "bde_norm": float(norm_bde),
                        "bdfe_norm": float(norm_bdfe),
                        "bde_missing": int(bde_missing),
                        "bdfe_missing": int(bdfe_missing),
                        "begin_is_som": int(begin_idx in soms),
                        "end_is_som": int(end_idx in soms),
                        "bond_adjacent_to_som": int(
                            (begin_idx in soms) or (end_idx in soms)
                        ),
                    }
                )

            self.add_bond_ranks(bond_records, value_col="bde_value", prefix="bde")
            self.add_bond_ranks(bond_records, value_col="bdfe_value", prefix="bdfe")

            for rec in bond_records:
                rec["is_bde_hot"] = int(
                    rec["bde_percentile_in_mol"] <= self.bde_hot_quantile
                )
                rec["is_bdfe_hot"] = int(
                    rec["bdfe_percentile_in_mol"] <= self.bde_hot_quantile
                )

            atom_records = self.build_atom_records(
                mol=mol,
                soms=soms,
                mol_id=mol_id,
                raw_mol_id=raw_mol_id,
                description=description,
                smiles=smiles,
                bond_records=bond_records,
            )

            x = torch.tensor(atom_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            y = torch.tensor(som_labels, dtype=torch.long)

            mol_ids = torch.full((len(atom_ids),), mol_id, dtype=torch.long)
            atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                mol_id=mol_ids,
                atom_id=atom_ids_tensor,
            )

            data.description = description
            data.smiles = smiles

            return data, atom_records, bond_records

        except Exception as e:
            print(f"[SOM] Error processing molecule {description}: {e}")
            return None, [], []

    @staticmethod
    def add_bond_ranks(
        records: List[Dict[str, Any]],
        value_col: str,
        prefix: str,
    ) -> None:
        n = len(records)
        if n == 0:
            return

        values = np.array([r[value_col] for r in records], dtype=float)
        order = np.argsort(values)

        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)

        if n == 1:
            percentiles = np.zeros(n, dtype=float)
        else:
            percentiles = (ranks - 1) / (n - 1)

        for i, r in enumerate(records):
            r[f"{prefix}_rank_in_mol"] = int(ranks[i])
            r[f"{prefix}_percentile_in_mol"] = float(percentiles[i])

    def build_atom_records(
        self,
        mol: Chem.Mol,
        soms: List[int],
        mol_id: int,
        raw_mol_id: int,
        description: str,
        smiles: str,
        bond_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_atom: Dict[int, List[Dict[str, Any]]] = {
            atom.GetIdx(): [] for atom in mol.GetAtoms()
        }

        for br in bond_records:
            by_atom[br["begin_atom_idx"]].append(br)
            by_atom[br["end_atom_idx"]].append(br)

        rows: List[Dict[str, Any]] = []

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            incident = by_atom.get(atom_idx, [])

            bde_vals = np.array([b["bde_value"] for b in incident], dtype=float)
            bdfe_vals = np.array([b["bdfe_value"] for b in incident], dtype=float)

            if len(incident) > 0:
                min_bde_idx = int(np.argmin(bde_vals))
                min_bdfe_idx = int(np.argmin(bdfe_vals))

                min_adj_bde = float(bde_vals[min_bde_idx])
                mean_adj_bde = float(np.mean(bde_vals))
                max_adj_bde = float(np.max(bde_vals))
                std_adj_bde = float(np.std(bde_vals))

                min_adj_bdfe = float(bdfe_vals[min_bdfe_idx])
                mean_adj_bdfe = float(np.mean(bdfe_vals))
                max_adj_bdfe = float(np.max(bdfe_vals))
                std_adj_bdfe = float(np.std(bdfe_vals))

                min_adj_bde_rank = incident[min_bde_idx]["bde_rank_in_mol"]
                min_adj_bde_percentile = incident[min_bde_idx][
                    "bde_percentile_in_mol"
                ]
                min_adj_bdfe_rank = incident[min_bdfe_idx]["bdfe_rank_in_mol"]
                min_adj_bdfe_percentile = incident[min_bdfe_idx][
                    "bdfe_percentile_in_mol"
                ]

                has_bde_hot_adjacent = int(any(b["is_bde_hot"] for b in incident))
                has_bdfe_hot_adjacent = int(any(b["is_bdfe_hot"] for b in incident))

                incident_bond_indices = ";".join(
                    str(b["bond_idx"]) for b in incident
                )

            else:
                min_adj_bde = mean_adj_bde = max_adj_bde = std_adj_bde = np.nan
                min_adj_bdfe = mean_adj_bdfe = max_adj_bdfe = std_adj_bdfe = np.nan
                min_adj_bde_rank = min_adj_bde_percentile = np.nan
                min_adj_bdfe_rank = min_adj_bdfe_percentile = np.nan
                has_bde_hot_adjacent = 0
                has_bdfe_hot_adjacent = 0
                incident_bond_indices = ""

            row: Dict[str, Any] = {
                "mol_id": mol_id,
                "raw_mol_id": raw_mol_id,
                "description": description,
                "smiles": smiles,
                "atom_idx": atom_idx,
                "orig_idx_1based": atom_idx + 1,
                "atom_symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "is_som": int(atom_idx in soms),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "total_degree": atom.GetTotalDegree(),
                "total_num_hs": atom.GetTotalNumHs(),
                "is_aromatic": int(atom.GetIsAromatic()),
                "is_ring": int(atom.IsInRing()),
                "hybridization": str(atom.GetHybridization()),
                "chiral_tag": str(atom.GetChiralTag()),
                "num_radical_electrons": atom.GetNumRadicalElectrons(),
                "neighbor_symbols": ";".join(
                    [n.GetSymbol() for n in atom.GetNeighbors()]
                ),
                "incident_bond_indices": incident_bond_indices,
                "num_incident_bonds": len(incident),
                "min_adj_bde": min_adj_bde,
                "mean_adj_bde": mean_adj_bde,
                "max_adj_bde": max_adj_bde,
                "std_adj_bde": std_adj_bde,
                "min_adj_bde_rank": min_adj_bde_rank,
                "min_adj_bde_percentile": min_adj_bde_percentile,
                "has_bde_hot_adjacent": has_bde_hot_adjacent,
                "min_adj_bdfe": min_adj_bdfe,
                "mean_adj_bdfe": mean_adj_bdfe,
                "max_adj_bdfe": max_adj_bdfe,
                "std_adj_bdfe": std_adj_bdfe,
                "min_adj_bdfe_rank": min_adj_bdfe_rank,
                "min_adj_bdfe_percentile": min_adj_bdfe_percentile,
                "has_bdfe_hot_adjacent": has_bdfe_hot_adjacent,
            }

            row.update(self.get_atom_class_flags(atom))
            rows.append(row)

        return rows

    @staticmethod
    def get_atom_class_flags(atom: Chem.Atom) -> Dict[str, int]:
        symbol = atom.GetSymbol()
        hyb = str(atom.GetHybridization())
        is_aromatic = atom.GetIsAromatic()
        is_ring = atom.IsInRing()
        num_h = atom.GetTotalNumHs()
        neighbor_symbols = [n.GetSymbol() for n in atom.GetNeighbors()]

        is_carbon_h = symbol == "C" and num_h > 0

        is_benzylic_or_allylic = False
        if is_carbon_h and hyb == "SP3":
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                if nbr.GetIsAromatic() or str(bond.GetBondType()) == "DOUBLE":
                    is_benzylic_or_allylic = True

        return {
            "class_aliphatic_c_h": int(
                is_carbon_h and (not is_aromatic) and hyb == "SP3"
            ),
            "class_benzylic_allylic_c_h": int(is_benzylic_or_allylic),
            "class_aromatic_c_h": int(is_carbon_h and is_aromatic),
            "class_heteroatom_adj_c_h": int(
                is_carbon_h
                and any(
                    s in ["N", "O", "S", "F", "Cl", "Br", "I"]
                    for s in neighbor_symbols
                )
            ),
            "class_c_n_adjacent": int("N" in neighbor_symbols),
            "class_c_o_adjacent": int("O" in neighbor_symbols),
            "class_c_s_adjacent": int("S" in neighbor_symbols),
            "class_ring_atom": int(is_ring),
            "class_sp2_atom": int(hyb == "SP2"),
            "class_sp3_atom": int(hyb == "SP3"),
        }

    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()

        element_list = [
            1,   # H
            5,   # B
            6,   # C
            7,   # N
            8,   # O
            9,   # F
            14,  # Si
            15,  # P
            16,  # S
            17,  # Cl
            35,  # Br
            53,  # I
        ]

        features = [1.0 if atomic_num == e else 0.0 for e in element_list]
        features.append(1.0 if atomic_num not in element_list else 0.0)

        return features

    def get_bond_features(
        self,
        bond: Chem.Bond,
        bde_val: float,
        bdfe_val: float,
    ) -> List[float]:
        bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        bond_type_str = str(bond.GetBondType())

        features = [1.0 if bond_type_str == t else 0.0 for t in bond_types]
        features.append(1.0 if bond_type_str not in bond_types else 0.0)

        features.append(1.0 if bond.IsInRing() else 0.0)
        features.append(1.0 if bond.GetIsConjugated() else 0.0)

        features.append(float(bde_val))
        features.append(float(bdfe_val))

        return features
