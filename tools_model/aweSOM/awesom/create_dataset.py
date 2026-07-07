import os
import shutil
from ast import literal_eval
from typing import Callable, List, Optional, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class SOM(InMemoryDataset):
    """
    PyTorch Geometric Dataset for site-of-metabolism prediction
    from SDF / SMI / SMILES files.

    Baseline no-BDE version:
        - remove hydrogens
        - atom features: element one-hot
        - bond features: bond type + ring + conjugation
        - no BDE/BDFE features
    """

    def __init__(
        self,
        root: str,
        labeled: bool = True,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], Data]] = None,
        force_reprocess: bool = True,
    ) -> None:
        self.labeled = labeled

        processed_folder = os.path.join(root, "processed")
        if force_reprocess and os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)
            print(f"Deleted existing processed folder at: {processed_folder}")

        super().__init__(root, transform, pre_transform, pre_filter)

        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
        input_file = self.find_input_file()

        if input_file is None:
            raise NotImplementedError(
                "Data file must be either .sdf, .smi, or .smiles."
            )

        data_list = self.data_processing(input_file=input_file)

        if len(data_list) == 0:
            raise RuntimeError("No valid molecules were processed.")

        torch.save(self.collate(data_list), self.processed_paths[0])

    def data_processing(self, input_file: str) -> List[Data]:
        _, file_extension = os.path.splitext(input_file)

        molecules, labels, descriptions = self.load_molecules(
            input_file,
            file_extension,
        )

        data_list = []

        for mol_id, (mol, soms, description) in enumerate(
            zip(molecules, labels, descriptions)
        ):
            if mol is None:
                continue

            mol, soms = self.remove_hydrogens_and_update_soms(mol, soms)

            if len(soms) == 0 and self.labeled:
                continue

            data = self.mol_to_data(mol, soms, mol_id, description)

            if data is not None:
                data_list.append(data)

        return data_list

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

            for mol in suppl:
                if mol is None:
                    continue

                soms = []
                if self.labeled:
                    soms_prop = mol.GetProp("soms") if mol.HasProp("soms") else "[]"
                    try:
                        soms = literal_eval(soms_prop)
                    except Exception:
                        soms = []

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
                        try:
                            soms = literal_eval(parts[2])
                        except Exception:
                            soms = []

                    desc = parts[1] if len(parts) > 1 else f"{line_num}"

                    molecules.append(mol)
                    labels.append(soms)
                    descriptions.append(desc)

        else:
            raise NotImplementedError(f"Invalid file extension: {file_extension}")

        return molecules, labels, descriptions

    def remove_hydrogens_and_update_soms(
        self,
        mol: Chem.Mol,
        soms: List[int],
    ) -> Tuple[Chem.Mol, List[int]]:
        """
        Remove hydrogens and update SOM indices.

        This keeps the original baseline behavior.
        """

        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            atom.SetIntProp("label", 1 if atom_id in soms else 0)

        mol_no_h = Chem.RemoveHs(mol)

        new_soms = []
        for atom in mol_no_h.GetAtoms():
            if atom.HasProp("label") and atom.GetIntProp("label") == 1:
                new_soms.append(atom.GetIdx())

        return mol_no_h, new_soms

    def mol_to_data(
        self,
        mol: Chem.Mol,
        soms: List[int],
        mol_id: int,
        description: str,
    ) -> Optional[Data]:
        try:
            atom_features = []
            atom_ids = []
            som_labels = []

            for atom in mol.GetAtoms():
                atom_id = atom.GetIdx()

                atom_features.append(self.get_atom_features(atom))
                atom_ids.append(atom_id)
                som_labels.append(1 if atom_id in soms else 0)

            edge_index_list = []
            edge_attr_list = []

            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()

                edge_index_list.append([begin_idx, end_idx])
                edge_attr_list.append(self.get_bond_features(bond))

            x = torch.tensor(atom_features, dtype=torch.float32)

            edge_index = torch.tensor(
                edge_index_list,
                dtype=torch.long,
            ).t().contiguous()

            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

            y = torch.tensor(som_labels, dtype=torch.long)

            mol_ids = torch.full(
                (len(atom_ids),),
                mol_id,
                dtype=torch.long,
            )

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

            return data

        except Exception as e:
            print(f"Error processing molecule {description}: {e}")
            return None

    def get_atom_features(self, atom: Chem.Atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()

        element_list = [
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

    def get_bond_features(self, bond: Chem.Bond) -> List[float]:
        bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        bond_type_str = str(bond.GetBondType())

        features = [1.0 if bond_type_str == t else 0.0 for t in bond_types]
        features.append(1.0 if bond_type_str not in bond_types else 0.0)

        features.append(1.0 if bond.IsInRing() else 0.0)
        features.append(1.0 if bond.GetIsConjugated() else 0.0)

        return features