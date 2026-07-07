import os

# 【强制补丁：彻底隔绝底层 C++ 线程池打架引发的死锁】
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import ast
import random
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

# 【强制补丁：限制 PyTorch 内部线程】
torch.set_num_threads(1)

import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

from e3nn import o3
import sys

# 动态添加当前脚本所在目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.AN_dataset import BDE2Dataset  # 保留原有依赖，便于独立脚本兼容
from model.DecNet_BDE2 import DecNet
from utils.MMAE import masked_mae  # 保留原有依赖，便于独立脚本兼容


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


class BDEPredictor:
    """
    常驻内存的 BDE 预测引擎。

    修改点：
    1. 修复 RDKit 版本不支持 params.maxAttempts 导致的全体预测失败问题；
    2. 增加 EmbedMultipleConfs + EmbedMolecule 双 fallback；
    3. MMFF 失败时 fallback 到 UFF；
    4. 增加 last_debug，方便外层判断到底在哪一步失败；
    5. 对 model output shape、edge index 映射、bond map 完整性做检查；
    6. 默认不打印大量正常预测结果，只在失败或 verbose=True 时打印。
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        mean=94.62390701792717,
        std=11.503478622893386,
        debug=False,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mean = mean
        self.std = std
        self.debug = debug
        self.last_debug = {}

        self.model = DecNet(
            order=2,
            basis_functions=128,
            cutoff=5.0,
            num_elements=1,
            avg_num_neighbors=3,
            correlation=3,
            num_interactions=6,
            heads=["dft"],
            hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"),
            MLP_irreps=o3.Irreps("64x0e"),
            gate=F.silu,
        )

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"[BDENET] checkpoint loaded: {checkpoint_path}", flush=True)
        print(f"[BDENET] device: {self.device}", flush=True)

    def _log(self, msg, verbose=False):
        if self.debug or verbose:
            print(msg, flush=True)

    def _warn(self, msg):
        print(msg, flush=True)

    def _reset_debug(self, mol):
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            smiles = "UNKNOWN"

        self.last_debug = {
            "smiles": smiles,
            "input_atoms": int(mol.GetNumAtoms()) if mol is not None else None,
            "input_bonds": int(mol.GetNumBonds()) if mol is not None else None,
            "failed_step": None,
            "error": None,
        }

    def _set_debug(self, **kwargs):
        self.last_debug.update(kwargs)

    def _generate_3d_conformer(self, mol, verbose=False):
        """
        RDKit 兼容版 3D 构象生成器。

        注意：
        新版/某些 RDKit 的 ETKDGv3 EmbedParameters 不支持 params.maxAttempts。
        因此不能直接写 params.maxAttempts = 50。
        """
        params = AllChem.ETKDGv3()
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        params.randomSeed = 42

        # 兼容不同 RDKit 版本：有这个属性才设置
        if hasattr(params, "maxAttempts"):
            try:
                params.maxAttempts = 50
            except Exception as exc:
                self._log(f"[BDENET][WARN] Cannot set params.maxAttempts: {repr(exc)}", verbose)

        if hasattr(params, "numThreads"):
            try:
                params.numThreads = 1
            except Exception:
                pass

        conf_id = None

        # 第一优先：EmbedMultipleConfs
        try:
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
            if ids:
                conf_id = int(ids[0])
        except Exception as exc:
            self._set_debug(
                failed_step="EmbedMultipleConfs",
                error=repr(exc),
            )
            self._log(f"[BDENET][WARN] EmbedMultipleConfs failed: {repr(exc)}", verbose)

        # 第二 fallback：EmbedMolecule，maxAttempts 作为函数参数传入
        if conf_id is None:
            try:
                cid = AllChem.EmbedMolecule(
                    mol,
                    maxAttempts=50,
                    randomSeed=42,
                    useExpTorsionAnglePrefs=True,
                    useBasicKnowledge=True,
                    enforceChirality=True,
                )

                if cid >= 0:
                    conf_id = int(cid)
            except Exception as exc:
                self._set_debug(
                    failed_step="EmbedMolecule",
                    error=repr(exc),
                )
                self._log(f"[BDENET][WARN] EmbedMolecule fallback failed: {repr(exc)}", verbose)

        if conf_id is None:
            self._set_debug(
                failed_step="generate_3d_conformer",
                error="cannot generate conformer",
            )
            self._warn("[BDENET][FAIL] 3D conformer generation failed.")
            return None

        # 优先 MMFF
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                status = AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant="MMFF94s",
                    confId=conf_id,
                    maxIters=200,
                )
                self._set_debug(mmff_status=int(status))
                return conf_id
        except Exception as exc:
            self._set_debug(
                mmff_error=repr(exc),
            )
            self._log(f"[BDENET][WARN] MMFF optimization failed: {repr(exc)}", verbose)

        # fallback UFF
        try:
            status = AllChem.UFFOptimizeMolecule(
                mol,
                confId=conf_id,
                maxIters=200,
            )
            self._set_debug(uff_status=int(status))
        except Exception as exc:
            self._set_debug(
                uff_error=repr(exc),
            )
            self._log(f"[BDENET][WARN] UFF optimization failed: {repr(exc)}", verbose)

        return conf_id

    def _prepare_pyg_data(self, mol_obj, verbose=False):
        """
        将分子转换为 PyG 图，并记录加氢前后的索引映射。
        返回:
            data: PyG Data
            edge_to_original_bond_map: {directed_edge_idx: original_bond_idx}
        """
        if mol_obj is None:
            self._set_debug(
                failed_step="input_mol",
                error="mol_obj is None",
            )
            return None, None

        original_num_atoms = int(mol_obj.GetNumAtoms())
        original_num_bonds = int(mol_obj.GetNumBonds())

        self._set_debug(
            original_num_atoms=original_num_atoms,
            original_num_bonds=original_num_bonds,
            original_num_conformers=int(mol_obj.GetNumConformers()),
        )

        if original_num_atoms == 0:
            self._set_debug(
                failed_step="input_mol",
                error="mol has zero atoms",
            )
            return None, None

        # 加氢并生成 3D
        try:
            if mol_obj.GetNumConformers() == 0:
                mol_3d = Chem.AddHs(mol_obj)
                best_conf_id = self._generate_3d_conformer(mol_3d, verbose=verbose)
            else:
                mol_3d = Chem.AddHs(mol_obj, addCoords=True)
                best_conf_id = 0
        except Exception as exc:
            self._set_debug(
                failed_step="AddHs_or_3D",
                error=repr(exc),
            )
            return None, None

        self._set_debug(
            atoms_after_add_hs=int(mol_3d.GetNumAtoms()),
            bonds_after_add_hs=int(mol_3d.GetNumBonds()),
            conformers_after_add_hs=int(mol_3d.GetNumConformers()),
            conf_id=int(best_conf_id) if best_conf_id is not None else None,
        )

        if best_conf_id is None:
            if self.last_debug.get("failed_step") is None:
                self._set_debug(
                    failed_step="generate_3d_conformer",
                    error="best_conf_id is None",
                )
            return None, None

        try:
            conf = mol_3d.GetConformer(int(best_conf_id))
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        except Exception as exc:
            self._set_debug(
                failed_step="GetConformer_or_positions",
                error=repr(exc),
            )
            return None, None

        atomic_number = [atom.GetAtomicNum() for atom in mol_3d.GetAtoms()]

        edge_feats = []
        edge_index = []
        edge_to_original_bond_map = {}

        current_edge_idx = 0

        for bond in mol_3d.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            edge_index.append([i, j])
            edge_index.append([j, i])

            # 只记录原始 heavy-atom bond，不记录新加 H 的 bond
            if i < original_num_atoms and j < original_num_atoms:
                original_bond = mol_obj.GetBondBetweenAtoms(i, j)
                if original_bond is not None:
                    original_bond_idx = int(original_bond.GetIdx())
                    edge_to_original_bond_map[current_edge_idx] = original_bond_idx
                    edge_to_original_bond_map[current_edge_idx + 1] = original_bond_idx

            bond_type = bond.GetBondType()

            if bond_type == Chem.rdchem.BondType.SINGLE:
                bond_type_val = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                bond_type_val = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                bond_type_val = 3
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_type_val = 4
            else:
                bond_type_val = 0

            edge_feat = float(bond_type_val + atomic_number[i] + atomic_number[j])
            edge_feats.extend([edge_feat, edge_feat])

            current_edge_idx += 2

        if not edge_index:
            self._set_debug(
                failed_step="build_edges",
                error="edge_index is empty",
            )
            return None, None

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        edge_feats_tensor = torch.tensor(edge_feats, dtype=torch.float32)
        z = torch.tensor(atomic_number, dtype=torch.float32)

        query_mask = torch.ones(edge_index_tensor.size(1), dtype=torch.bool)

        mapped_original_bonds = sorted(set(edge_to_original_bond_map.values()))
        missing_original_bonds = [
            idx for idx in range(original_num_bonds)
            if idx not in set(mapped_original_bonds)
        ]

        self._set_debug(
            num_directed_edges=int(edge_index_tensor.size(1)),
            num_edge_feats=int(edge_feats_tensor.numel()),
            num_index_map_entries=int(len(edge_to_original_bond_map)),
            mapped_original_bonds=mapped_original_bonds,
            missing_original_bonds_before_model=missing_original_bonds,
        )

        self._log(
            "[BDENET][DATA] "
            f"orig_atoms={original_num_atoms}, orig_bonds={original_num_bonds}, "
            f"atoms_H={mol_3d.GetNumAtoms()}, bonds_H={mol_3d.GetNumBonds()}, "
            f"directed_edges={edge_index_tensor.size(1)}, "
            f"mapped_bonds={len(mapped_original_bonds)}, "
            f"missing_before_model={missing_original_bonds[:20]}",
            verbose,
        )

        data = Data(
            z=z,
            pos=pos,
            edge_index=edge_index_tensor,
            edge_feats=edge_feats_tensor,
            query_mask=query_mask,
            molecule_size=len(pos),
        )

        return data, edge_to_original_bond_map

    @torch.no_grad()
    def predict(self, mol: Chem.Mol, verbose=False) -> dict:
        """
        在线推理接口。
        输入:
            mol: RDKit Mol
        返回:
            {原始 bond_idx: predicted_bde}
        """
        self._reset_debug(mol)

        if mol is None:
            self._set_debug(
                failed_step="input_mol",
                error="mol is None",
            )
            return {}

        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            smiles = "UNKNOWN"

        self._log(f"[BDENET] predict start: {smiles}", verbose)

        try:
            data, index_map = self._prepare_pyg_data(mol, verbose=verbose)
        except Exception as exc:
            self._set_debug(
                failed_step="_prepare_pyg_data_exception",
                error=repr(exc),
            )
            self._warn(f"[BDENET][FAIL] _prepare_pyg_data exception on {smiles}: {repr(exc)}")
            return {}

        if data is None or index_map is None:
            if self.last_debug.get("failed_step") is None:
                self._set_debug(
                    failed_step="_prepare_pyg_data",
                    error="data or index_map is None",
                )

            self._warn(
                f"[BDENET][FAIL] prepare data failed on {smiles}: "
                f"step={self.last_debug.get('failed_step')} "
                f"error={self.last_debug.get('error')}"
            )
            return {}

        try:
            data.batch = torch.zeros(data.z.size(0), dtype=torch.long)
            data = data.to(self.device)
        except Exception as exc:
            self._set_debug(
                failed_step="data_to_device",
                error=repr(exc),
            )
            self._warn(f"[BDENET][FAIL] data.to(device) failed on {smiles}: {repr(exc)}")
            return {}

        try:
            raw_output = self.model(data, self.mean, self.std)
        except Exception as exc:
            self._set_debug(
                failed_step="model_forward",
                error=repr(exc),
            )
            self._warn(f"[BDENET][FAIL] model forward failed on {smiles}: {repr(exc)}")
            return {}

        try:
            bde_pred = raw_output.float().detach().cpu().numpy()
            bde_pred = np.asarray(bde_pred).reshape(-1)
        except Exception as exc:
            self._set_debug(
                failed_step="output_to_numpy",
                error=repr(exc),
            )
            self._warn(f"[BDENET][FAIL] output conversion failed on {smiles}: {repr(exc)}")
            return {}

        n_directed_edges = int(data.edge_index.size(1))
        n_pred = int(len(bde_pred))

        self._set_debug(
            model_output_shape=list(raw_output.shape) if hasattr(raw_output, "shape") else str(type(raw_output)),
            n_pred=n_pred,
            n_directed_edges=n_directed_edges,
        )

        self._log(
            f"[BDENET][MODEL] output_shape={getattr(raw_output, 'shape', None)}, "
            f"n_pred={n_pred}, n_directed_edges={n_directed_edges}",
            verbose,
        )

        if n_pred < n_directed_edges:
            self._set_debug(
                failed_step="model_output_length",
                error=f"model output shorter than directed edges: n_pred={n_pred}, n_edges={n_directed_edges}",
            )
            self._warn(
                f"[BDENET][FAIL] model output shorter than directed edges on {smiles}: "
                f"n_pred={n_pred}, n_edges={n_directed_edges}"
            )
            return {}

        bond_bde_map = {}

        for edge_idx, original_bond_idx in index_map.items():
            edge_idx = int(edge_idx)
            original_bond_idx = int(original_bond_idx)

            # 每个 undirected bond 有两个 directed edges，只取偶数边和后一条反向边平均
            if edge_idx % 2 != 0:
                continue

            reverse_edge_idx = edge_idx + 1

            if reverse_edge_idx >= n_pred:
                self._log(
                    f"[BDENET][WARN] reverse edge out of range: "
                    f"edge={edge_idx}, reverse={reverse_edge_idx}, n_pred={n_pred}",
                    verbose,
                )
                continue

            avg_bde = (float(bde_pred[edge_idx]) + float(bde_pred[reverse_edge_idx])) / 2.0
            bond_bde_map[original_bond_idx] = float(avg_bde)

        original_bonds = int(mol.GetNumBonds())
        predicted_keys = sorted(bond_bde_map.keys())
        missing_after_model = [
            idx for idx in range(original_bonds)
            if idx not in bond_bde_map
        ]

        values = list(bond_bde_map.values())

        self._set_debug(
            n_predicted_original_bonds=int(len(bond_bde_map)),
            predicted_original_bond_keys=predicted_keys,
            missing_original_bonds_after_model=missing_after_model,
            bde_min=float(min(values)) if values else None,
            bde_mean=float(sum(values) / len(values)) if values else None,
        )

        if not bond_bde_map:
            self._set_debug(
                failed_step="bond_bde_map_empty",
                error="no original bond received BDE prediction",
            )
            self._warn(f"[BDENET][FAIL] bond_bde_map empty on {smiles}")
            return {}

        if missing_after_model:
            self._warn(
                f"[BDENET][WARN] incomplete original bond prediction on {smiles}: "
                f"predicted={len(bond_bde_map)}/{original_bonds}, "
                f"missing={missing_after_model[:30]}"
            )

        self._log(
            f"[BDENET][OK] predicted_bonds={len(bond_bde_map)}/{original_bonds}, "
            f"min={min(values):.3f}, mean={sum(values) / len(values):.3f}",
            verbose,
        )

        return bond_bde_map


# ==========================================
# 下方为原有的批量处理逻辑，保持不变，便于作为独立脚本运行
# ==========================================

def parse_input_data(input_arg, mode):
    raw_data = []

    if os.path.isfile(input_arg):
        if input_arg.endswith(".csv"):
            df = pd.read_csv(input_arg)

            for _, row in df.iterrows():
                smiles = row["molecule"]
                mol = Chem.MolFromSmiles(smiles)

                bond_index = None
                if "bond_index" in row and pd.notna(row["bond_index"]):
                    bond_index = (
                        ast.literal_eval(row["bond_index"])
                        if isinstance(row["bond_index"], str)
                        else row["bond_index"]
                    )

                item = {
                    "mol_obj": mol,
                    "smiles": smiles,
                    "bond_index": bond_index,
                }

                if mode == "test" and "bde" in row:
                    item["bde_vals"] = (
                        ast.literal_eval(row["bde"])
                        if isinstance(row["bde"], str)
                        else row["bde"]
                    )

                raw_data.append(item)

        elif input_arg.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(input_arg, removeHs=False)

            for mol in suppl:
                if mol is None:
                    continue

                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                bond_idx_str = mol.GetProp("bond_index") if mol.HasProp("bond_index") else None
                bde_str = mol.GetProp("bde") if mol.HasProp("bde") else None

                bond_index = ast.literal_eval(bond_idx_str) if bond_idx_str else None

                item = {
                    "mol_obj": mol,
                    "smiles": smiles,
                    "bond_index": bond_index,
                }

                if mode == "test" and bde_str:
                    item["bde_vals"] = ast.literal_eval(bde_str)

                raw_data.append(item)

    else:
        mol = Chem.MolFromSmiles(input_arg)
        if mol is not None:
            raw_data.append(
                {
                    "mol_obj": mol,
                    "smiles": input_arg,
                    "bond_index": None,
                }
            )

    return raw_data


def run_batch_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    set_seed(2021)

    predictor = BDEPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        mean=args.mean,
        std=args.std,
        debug=args.debug,
    )

    raw_data_list = parse_input_data(args.input, args.mode)
    results = []

    print(f"Running batch inference in {args.mode.upper()} mode...", flush=True)
    start_time = time()

    for item in tqdm(raw_data_list, desc="Processing Molecules", unit="mol"):
        mol = item["mol_obj"]

        if mol is None:
            continue

        bde_map = predictor.predict(mol, verbose=args.verbose)

        smiles = item["smiles"]
        target_bonds = item["bond_index"] if item["bond_index"] is not None else list(bde_map.keys())

        if target_bonds is None:
            target_bonds = []

        for b_idx in target_bonds:
            try:
                b_idx = int(b_idx)
            except Exception:
                continue

            if b_idx in bde_map:
                bond = mol.GetBondWithIdx(b_idx)

                res = {
                    "SMILES": smiles,
                    "Bond_Idx": b_idx,
                    "Atom1_Idx": bond.GetBeginAtomIdx(),
                    "Atom2_Idx": bond.GetEndAtomIdx(),
                    "Prediction_BDE": bde_map[b_idx],
                }

                if item.get("bde_vals") is not None and b_idx < len(item["bde_vals"]):
                    res["True_BDE"] = item["bde_vals"][b_idx]

                results.append(res)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_csv, index=False)

    print(
        f"Results successfully saved to CSV: {args.output_csv} | "
        f"Time: {time() - start_time:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDE Prediction Pipeline")

    parser.add_argument("--input", type=str, required=True, help="Input source.")
    parser.add_argument("--mode", type=str, choices=["test", "predict"], default="predict")
    parser.add_argument("--output_csv", type=str, default="bde_predictions.csv")
    parser.add_argument("--output_sdf", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=current_dir + "/checkpoints/DecNet/bde_checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mean", type=float, default=94.62390701792717)
    parser.add_argument("--std", type=float, default=11.503478622893386)

    # 新增调试参数
    parser.add_argument("--debug", action="store_true", help="Print detailed BDENET debug information.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose prediction logs.")

    args = parser.parse_args()
    run_batch_inference(args)