import argparse
import ast
import random
import os
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from e3nn import o3
import sys

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.AN_dataset import BDE2Dataset
from model.DecNet_BDE2 import DecNet
from utils.MMAE import masked_mae

torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def load_model(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    return model

def generate_3d_conformer(mol):
    max_attempts = 10
    rmsd_threshold = 0.5
    num_confs = 10

    params = AllChem.ETKDGv3()
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

    success = False
    for attempt in range(max_attempts):
        params.randomSeed = attempt
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if ids:
            success = True
            break
            
    if not success:
        raise ValueError("Failed to generate 3D conformer.")

    for conf_id in ids:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    energies = [AllChem.UFFGetMoleculeForceField(mol, confId=conf).CalcEnergy() for conf in ids]
    lowest_energy_conf = energies.index(min(energies))

    unique_confs = [lowest_energy_conf]
    for i in ids:
        if all(rdMolAlign.GetBestRMS(mol, mol, i, j) > rmsd_threshold for j in unique_confs):
            unique_confs.append(i)

    for conf_id in unique_confs:
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    return lowest_energy_conf


def mol_to_graph(input_dict, mode='predict'):
    mol = input_dict['mol_obj']
    smiles = input_dict['smiles']
    bde_index = input_dict['bond_index']
    
    bde_vals = input_dict.get('bde_vals', None)
    bdfe_vals = input_dict.get('bdfe_vals', None)

    if mol is None: return None

    try:
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            best_conf_id = generate_3d_conformer(mol)
        else:
            mol = Chem.AddHs(mol, addCoords=True)
            best_conf_id = 0
            
        pos = torch.tensor(mol.GetConformer(best_conf_id).GetPositions(), dtype=torch.float32)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_feats = []
    edge_index = []

    for bond_idx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE: bond_type_val = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE: bond_type_val = 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE: bond_type_val = 3
        elif bond_type == Chem.rdchem.BondType.AROMATIC: bond_type_val = 4
        else: bond_type_val = 0

        src_type = atomic_number[i]
        dst_type = atomic_number[j]

        edge_feat = torch.tensor(bond_type_val + src_type + dst_type, dtype=torch.float32)
        edge_feats.extend([edge_feat, edge_feat])

    if bde_index is None:
        bde_index = list(range(mol.GetNumBonds()))

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T
    z = torch.tensor(atomic_number, dtype=torch.float32)
    edge_size = edge_index_tensor.size(1)

    bde_tensor = torch.full((edge_size,), float('nan'))
    bdfe_tensor = torch.full((edge_size,), float('nan'))

    for idx, b_idx in enumerate(bde_index):
        if mode == 'test':
            if bde_vals is not None and idx < len(bde_vals):
                bde_tensor[2 * b_idx] = bde_vals[idx]
                bde_tensor[2 * b_idx + 1] = bde_vals[idx]
            if bdfe_vals is not None and idx < len(bdfe_vals):
                bdfe_tensor[2 * b_idx] = bdfe_vals[idx]
                bdfe_tensor[2 * b_idx + 1] = bdfe_vals[idx]

    mol_block = Chem.MolToMolBlock(mol, confId=best_conf_id)

    data = Data(z=z, pos=pos, edge_index=edge_index_tensor, edge_feats=torch.tensor(edge_feats),
                bde_index=torch.tensor(bde_index, dtype=torch.long), 
                bde=bde_tensor, bdfe=bdfe_tensor,
                molecule_size=len(pos), smiles=smiles, mol_block=mol_block)
    return data


def parse_input_data(input_arg, mode):
    raw_data = []
    if os.path.isfile(input_arg):
        if input_arg.endswith('.csv'):
            print(f"Detected CSV input. Grouping by molecule...")
            df = pd.read_csv(input_arg)
            
            smiles_col = 'molecule' if 'molecule' in df.columns else 'smiles'
            if smiles_col not in df.columns:
                raise ValueError("CSV must contain a 'molecule' or 'smiles' column.")

            def safe_eval(val):
                if isinstance(val, str):
                    try: return ast.literal_eval(val)
                    except: return val
                return val
            
            for col in ['bond_index', 'bde', 'bdfe']:
                if col in df.columns: df[col] = df[col].apply(safe_eval)

            def flatten_or_collect(series):
                result = []
                for item in series.dropna():
                    if isinstance(item, list): result.extend(item)
                    else: result.append(item)
                return result if result else None

            agg_dict = {}
            if 'bond_index' in df.columns: agg_dict['bond_index'] = flatten_or_collect
            if mode == 'test':
                if 'bde' in df.columns: agg_dict['bde'] = flatten_or_collect
                if 'bdfe' in df.columns: agg_dict['bdfe'] = flatten_or_collect
                
            grouped_df = df.groupby(smiles_col).agg(agg_dict).reset_index()
            print(f"Optimization: Compressed {len(df)} rows into {len(grouped_df)} unique 3D molecules.")

            for _, row in grouped_df.iterrows():
                smiles = row[smiles_col]
                item = {
                    'mol_obj': Chem.MolFromSmiles(smiles),
                    'smiles': smiles,
                    'bond_index': row.get('bond_index', None)
                }
                if mode == 'test':
                    if 'bde' in row: item['bde_vals'] = row['bde']
                    if 'bdfe' in row: item['bdfe_vals'] = row['bdfe']
                raw_data.append(item)
                
        elif input_arg.endswith('.sdf'):
            print(f"Detected SDF input.")
            suppl = Chem.SDMolSupplier(input_arg, removeHs=False)
            for mol in suppl:
                if mol is None: continue
                smiles = Chem.MolToSmiles(mol)
                bond_idx_str = mol.GetProp('bond_index') if mol.HasProp('bond_index') else None
                item = {
                    'mol_obj': mol,
                    'smiles': smiles,
                    'bond_index': ast.literal_eval(bond_idx_str) if bond_idx_str else None
                }
                if mode == 'test':
                    if mol.HasProp('bde'): item['bde_vals'] = ast.literal_eval(mol.GetProp('bde'))
                    if mol.HasProp('bdfe'): item['bdfe_vals'] = ast.literal_eval(mol.GetProp('bdfe'))
                raw_data.append(item)
        else: raise ValueError("Unsupported file format.")
    else:
        print(f"Detected single SMILES input: {input_arg}")
        mol = Chem.MolFromSmiles(input_arg)
        if mol is None: raise ValueError("Invalid SMILES.")
        raw_data.append({'mol_obj': mol, 'smiles': input_arg, 'bond_index': None })
        
    return raw_data


def evaluate_metrics(trues, preds, target_name):
    if len(trues) == 0: return
    trues_arr, preds_arr = np.array(trues), np.array(preds)
    mae = np.mean(np.abs(trues_arr - preds_arr))
    r2 = r2_score(trues_arr, preds_arr)
    pearson_corr, _ = pearsonr(trues_arr, preds_arr)
    spearman_corr, _ = spearmanr(trues_arr, preds_arr)
    
    print(f"\n{'='*40}\n {target_name.upper()} EVALUATION METRICS \n{'='*40}")
    print(f" Number of Bonds : {len(trues_arr)}\n MAE             : {mae:.4f} kcal/mol")
    print(f" R-squared (R2)  : {r2:.4f}\n Pearson (r)     : {pearson_corr:.4f}")
    print(f" Spearman (rho)  : {spearman_corr:.4f}\n{'='*40}")


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Target: BOTH BDE AND BDFE")
    set_seed(2026)

    print("Loading and parsing input data...")
    raw_data_list = parse_input_data(args.input, args.mode)

    print("Processing graphs and generating 3D conformers...")
    data_list = Parallel(n_jobs=-1)(delayed(mol_to_graph)(item, args.mode) for item in tqdm(raw_data_list))
    data_list = [d for d in data_list if d is not None]

    if not data_list: return print("Error: No valid molecules could be processed.")

    dataset = BDE2Dataset(data_list)
    loader = PyGDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model_kwargs = dict(
        order=2, basis_functions=128, cutoff=5.0, num_elements=1,
        avg_num_neighbors=3, correlation=3, num_interactions=6, heads=["dft"],
        hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"), MLP_irreps=o3.Irreps("64x0e"), gate=F.silu
    )

    print("Loading BDE Model...")
    model_bde = DecNet(**model_kwargs)
    try: load_model(model_bde, args.ckpt_bde, device)
    except FileNotFoundError: return print(f"Error: Checkpoint not found at {args.ckpt_bde}.")
    model_bde.to(device).eval()

    print("Loading BDFE Model...")
    model_bdfe = DecNet(**model_kwargs)
    try: load_model(model_bdfe, args.ckpt_bdfe, device)
    except FileNotFoundError: return print(f"Error: Checkpoint not found at {args.ckpt_bdfe}.")
    model_bdfe.to(device).eval()

    results = []
    print(f"Running inference in {args.mode.upper()} mode...")
    start_time = time()

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            
            bde_preds = model_bde(batch, args.mean_bde, args.std_bde).float()
            bdfe_preds = model_bdfe(batch, args.mean_bdfe, args.std_bdfe).float()
            
            graphs = batch.to_data_list()
            edge_counts = [g.num_edges for g in graphs]
            bde_preds_split = torch.split(bde_preds, edge_counts)
            bdfe_preds_split = torch.split(bdfe_preds, edge_counts)

            for g_idx, g in enumerate(graphs):
                smiles = g.smiles[0] if isinstance(g.smiles, list) else g.smiles
                mol_block = g.mol_block[0] if isinstance(g.mol_block, list) else g.mol_block
                
                g_bde_preds = bde_preds_split[g_idx]
                g_bdfe_preds = bdfe_preds_split[g_idx]
                
                if not hasattr(g, 'bde_index') or g.bde_index is None: continue
                bde_indices = g.bde_index.cpu().numpy().flatten()
                
                for bond_idx in bde_indices:
                    # 原生取均值：键 bond_idx 完美对应有向边 2*bond_idx 和 2*bond_idx+1
                    avg_bde = (g_bde_preds[2 * bond_idx] + g_bde_preds[2 * bond_idx + 1]).item() / 2.0
                    avg_bdfe = (g_bdfe_preds[2 * bond_idx] + g_bdfe_preds[2 * bond_idx + 1]).item() / 2.0

                    u = g.edge_index[0, 2 * bond_idx].item()
                    v = g.edge_index[1, 2 * bond_idx].item()
                    z_u = int(g.z[u].item())
                    z_v = int(g.z[v].item())

                    true_bde = g.bde[2 * bond_idx].item() if hasattr(g, 'bde') else np.nan
                    true_bdfe = g.bdfe[2 * bond_idx].item() if hasattr(g, 'bdfe') else np.nan

                    results.append({
                        'smiles': smiles,
                        'bond_index': bond_idx,
                        'atom_type1': z_u,
                        'atom_type2': z_v,
                        'bde_label': true_bde,
                        'bdfe_label': true_bdfe,
                        'bde_pred': avg_bde,
                        'bdfe_pred': avg_bdfe,
                        '_local_u': u,
                        '_local_v': v,
                        'Mol_Block': mol_block
                    })

    print(f"\nInference Time: {time() - start_time:.2f} s")
    df_out = pd.DataFrame(results)

    if args.mode == 'test':
        df_eval_bde = df_out.dropna(subset=['bde_label'])
        evaluate_metrics(df_eval_bde['bde_label'].tolist(), df_eval_bde['bde_pred'].tolist(), "BDE")
        df_eval_bdfe = df_out.dropna(subset=['bdfe_label'])
        evaluate_metrics(df_eval_bdfe['bdfe_label'].tolist(), df_eval_bdfe['bdfe_pred'].tolist(), "BDFE")

    csv_columns = ['smiles', 'bond_index', 'atom_type1', 'atom_type2', 'bde_label', 'bdfe_label', 'bde_pred', 'bdfe_pred']
    df_csv = df_out[csv_columns]
    df_csv.to_csv(args.output_csv, index=False)
    print(f"\nResults successfully saved to CSV: {args.output_csv}")

    if args.output_sdf:
        print(f"Generating SDF file with BDE and BDFE Summaries...")
        writer = Chem.SDWriter(args.output_sdf)
        grouped = df_out.groupby('smiles')
        for smiles, group in grouped:
            mol_block = group.iloc[0]['Mol_Block']
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
            
            summary_pred_bde, summary_pred_bdfe = [], []
            summary_true_bde, summary_true_bdfe = [], []
            
            for _, row in group.iterrows():
                atom1, atom2 = int(row['_local_u']) + 1, int(row['_local_v']) + 1
                summary_pred_bde.append(f"{atom1}-{atom2}: {row['bde_pred']:.2f}")
                summary_pred_bdfe.append(f"{atom1}-{atom2}: {row['bdfe_pred']:.2f}")

                if args.mode == 'test':
                    if pd.notna(row['bde_label']): summary_true_bde.append(f"{atom1}-{atom2}: {row['bde_label']:.2f}")
                    if pd.notna(row['bdfe_label']): summary_true_bdfe.append(f"{atom1}-{atom2}: {row['bdfe_label']:.2f}")

            mol.SetProp("Prediction_BDE_Summary", "\n".join(summary_pred_bde))
            mol.SetProp("Prediction_BDFE_Summary", "\n".join(summary_pred_bdfe))
            if summary_true_bde: mol.SetProp("True_BDE_Summary", "\n".join(summary_true_bde))
            if summary_true_bdfe: mol.SetProp("True_BDFE_Summary", "\n".join(summary_true_bdfe))
            mol.SetPro("_Name", smiles) 
            writer.write(mol)
            
        writer.close()
        print(f"SDF file successfully saved to: {args.output_sdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDE and BDFE Simultaneous Prediction Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input source (.csv, .sdf, or SMILES)')
    parser.add_argument('--mode', type=str, choices=['test', 'predict'], default='predict')
    parser.add_argument('--output_csv', type=str, default='BDENET_extra.csv')
    parser.add_argument('--output_sdf', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    
    args.mean_bde, args.std_bde = 94.62390701792717, 11.503478622893386
    args.ckpt_bde = os.path.join(current_dir, 'checkpoints', 'BDE_ckpt_new', 'bde_checkpoint')
    
    args.mean_bdfe, args.std_bdfe = 84.48772566704848, 12.50514247866427
    args.ckpt_bdfe = os.path.join(current_dir, 'checkpoints', 'BDE_ckpt_new', 'bdfe_checkpoint')
    
    run_inference(args)
