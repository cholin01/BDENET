#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced reaction-center test script.

Compared with the original test_center.py, this version keeps the original
evaluation logic but additionally exports analysis-ready CSV files for
Figure-style analysis:

Per model run:
    center_reaction_level_<tag>.csv
    center_result_long_<tag>.csv
    center_topk_summary_<tag>.csv
    center_batch_progress_<tag>.csv
    center_skipped_<tag>.csv
    center_csv_manifest_<tag>.csv

Optional comparison mode after running baseline and +BDE separately:
    center_reaction_comparison.csv
    center_topk_comparison.csv
    center_case_type_summary.csv
    center_rank_shift_summary.csv

Candidate-level output when used with the modified molcenter.py:
    center_candidate_table_<tag>.csv
    center_bde_enrichment_<tag>.csv
These contain candidate score/rank, atom/bond identity, true-center label, and BDE features.
"""

import os
import sys
import time
import math
import random
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors

from chemutils import remove_atommap, is_sim
from mol_tree import MolTree
from molcenter_photo import MolCenter
from vocab import Vocab, common_atom_vocab
from datautils import MolTreeFolder

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_tree(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol_tree = MolTree(smiles)
    return mol_tree


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float_array(x, knum: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < knum:
        out = np.zeros(knum, dtype=float)
        out[: arr.size] = arr
        return out
    return arr[:knum]


def first_hit_rank(hit_vec: Iterable[float], miss_rank: Optional[int] = None) -> float:
    vals = list(hit_vec)
    for i, v in enumerate(vals, start=1):
        try:
            if float(v) >= 0.5:
                return float(i)
        except Exception:
            continue
    return float(miss_rank if miss_rank is not None else len(vals) + 1)


def parse_test_file(test_path: str) -> Tuple[List[Tuple[int, str, str, str]], pd.DataFrame]:
    """
    Parse the same test CSV expected by the original script.

    Original format assumption:
        col0: integer input row id
        col1: reaction/patent id
        col2: reactants>>product reaction SMILES

    Extra columns, if present, are retained in metadata as extra_col_3, ...
    """
    data = []
    meta_rows = []

    with open(test_path) as f:
        lines = f.readlines()

    header = []
    if len(lines) > 0:
        header = [h.strip() for h in lines[0].strip("\r\n ").split(",")]

    for line_no, line in enumerate(lines[1:], start=2):
        line = line.strip("\r\n ")
        if not line:
            continue

        # Keep the original split behavior for compatibility.
        s = line.split(",")
        if len(s) < 3:
            continue

        try:
            input_row_id = int(s[0])
        except Exception:
            # Fallback: preserve order if the first field is not int.
            input_row_id = len(data)

        sample_id = str(s[1])
        rxn_smiles = s[2]
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            # Invalid reaction smiles; still keep row for traceability.
            react_smiles, product_smiles = "", rxn_smiles
        else:
            react_smiles, product_smiles = parts[0], parts[1]

        data.append((input_row_id, sample_id, product_smiles, react_smiles))

        row = {
            "input_row_id": input_row_id,
            "sample_id": sample_id,
            "reaction_smiles": rxn_smiles,
            "reactant_smiles": react_smiles,
            "product_smiles_from_file": product_smiles,
            "line_no": line_no,
        }
        for j in range(3, len(s)):
            if j < len(header) and header[j]:
                col_name = header[j]
            else:
                col_name = f"extra_col_{j}"
            # Avoid overwriting required canonical columns.
            if col_name in row:
                col_name = f"extra_col_{j}"
            row[col_name] = s[j]
        meta_rows.append(row)

    return data, pd.DataFrame(meta_rows)


def build_meta_lookup(meta_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if meta_df.empty or "sample_id" not in meta_df.columns:
        return {}
    out = {}
    for _, row in meta_df.iterrows():
        out[str(row["sample_id"])] = row.to_dict()
    return out


def call_validate_centers(
    model,
    classes,
    product_batch,
    product_tree,
    knum: int,
    sample_ids: Optional[List[str]] = None,
    product_smiles_list: Optional[List[str]] = None,
):
    """
    Try to obtain candidate-level details from the modified MolCenter.

    New MolCenter API:
        validate_centers(..., return_details=True, sample_ids=..., product_smiles_list=...)

    Fallback to the original API if the local MolCenter has not been replaced.
    """
    details = None

    try:
        result = model.validate_centers(
            classes,
            product_batch,
            product_tree,
            [],
            knum=knum,
            return_details=True,
            sample_ids=sample_ids,
            product_smiles_list=product_smiles_list,
        )
        if isinstance(result, tuple) and len(result) >= 4:
            top_5_acc, bond_center_acc, atom_center_acc, details = result[0], result[1], result[2], result[3]
        else:
            top_5_acc, bond_center_acc, atom_center_acc = result
    except TypeError:
        top_5_acc, bond_center_acc, atom_center_acc = model.validate_centers(
            classes,
            product_batch,
            product_tree,
            [],
            knum=knum,
        )

    return top_5_acc, bond_center_acc, atom_center_acc, details


def normalize_candidate_details(details: Any, model_tag: str, batch_idx: int) -> pd.DataFrame:
    """
    Convert optional candidate details into a DataFrame.

    Supported loose formats:
        list[dict]
        dict[str, list/array]
        pandas.DataFrame

    The original MolCenter.validate_centers usually does not expose details.
    In that case this returns an empty DataFrame.
    """
    if details is None:
        return pd.DataFrame()

    if isinstance(details, pd.DataFrame):
        df = details.copy()
    elif isinstance(details, list):
        if len(details) == 0:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in details):
            df = pd.DataFrame(details)
        else:
            return pd.DataFrame()
    elif isinstance(details, dict):
        try:
            df = pd.DataFrame(details)
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    df["model_tag"] = model_tag
    df["batch_idx"] = batch_idx
    return df


def make_topk_summary_from_reaction_rows(rows: pd.DataFrame, model_tag: str, knum: int) -> pd.DataFrame:
    out = []

    for include_skipped, name in [(True, "overall_including_skipped"), (False, "overall_eval_only")]:
        sub = rows.copy()
        if not include_skipped and "skipped" in sub.columns:
            sub = sub[sub["skipped"] == 0].copy()

        for k in range(1, knum + 1):
            col = f"top{k}_hit"
            if col not in sub.columns or len(sub) == 0:
                acc = np.nan
                n = 0
            else:
                acc = float(pd.to_numeric(sub[col], errors="coerce").fillna(0).mean())
                n = int(len(sub))
            out.append(
                {
                    "model_tag": model_tag,
                    "center_type": name,
                    "k": k,
                    "accuracy": acc,
                    "n": n,
                }
            )

    return pd.DataFrame(out)


def make_subset_topk_summary(acc_array: Optional[np.ndarray], model_tag: str, center_type: str, knum: int) -> pd.DataFrame:
    rows = []
    if acc_array is None:
        for k in range(1, knum + 1):
            rows.append({"model_tag": model_tag, "center_type": center_type, "k": k, "accuracy": np.nan, "n": 0})
        return pd.DataFrame(rows)

    arr = np.asarray(acc_array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n = int(arr.shape[0])
    for k in range(1, knum + 1):
        if n == 0 or arr.shape[1] < k:
            acc = np.nan
        else:
            acc = float(np.nanmean(arr[:, k - 1]))
        rows.append({"model_tag": model_tag, "center_type": center_type, "k": k, "accuracy": acc, "n": n})

    return pd.DataFrame(rows)


def make_true_center_lookup(candidate_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Extract one true-center row per sample from candidate details.
    Used to enrich the reaction-level CSV with true-center type and BDE fields.
    """
    if candidate_df is None or candidate_df.empty or "sample_id" not in candidate_df.columns:
        return {}
    if "is_true_center" not in candidate_df.columns:
        return {}

    true_df = candidate_df[pd.to_numeric(candidate_df["is_true_center"], errors="coerce").fillna(0) == 1].copy()
    lookup = {}
    for sid, g in true_df.groupby("sample_id"):
        r = g.iloc[0].to_dict()
        lookup[str(sid)] = {
            "true_center_type": r.get("true_center_type", r.get("candidate_type", np.nan)),
            "true_candidate_type": r.get("candidate_type", np.nan),
            "true_candidate_label": r.get("candidate_label", np.nan),
            "true_candidate_rank": r.get("candidate_rank", np.nan),
            "true_candidate_atom_i": r.get("candidate_atom_i", np.nan),
            "true_candidate_atom_j": r.get("candidate_atom_j", np.nan),
            "true_candidate_bond_local_idx": r.get("candidate_bond_local_idx", np.nan),
            "true_candidate_bde": r.get("candidate_bde", np.nan),
            "true_candidate_bde_percentile_in_mol": r.get("candidate_bde_percentile_in_mol", np.nan),
            "true_candidate_min_adjacent_bde": r.get("candidate_min_adjacent_bde", np.nan),
            "true_candidate_min_adjacent_bde_percentile": r.get("candidate_min_adjacent_bde_percentile", np.nan),
        }
    return lookup


def make_bde_enrichment_from_candidates(candidate_df: pd.DataFrame, out_csv: str, n_bins: int = 5) -> pd.DataFrame:
    """
    True reaction-center enrichment by BDE percentile.

    Uses unique candidate bonds rather than the four duplicated bond-change labels.
    Atom centers are summarized by min adjacent BDE percentile when available.
    """
    if candidate_df is None or candidate_df.empty:
        out = pd.DataFrame()
        out.to_csv(out_csv, index=False)
        return out

    rows = []

    # Bond-center enrichment: unique bond per molecule.
    if {"candidate_type", "sample_id", "candidate_bond_local_idx", "candidate_bde_percentile_in_mol"}.issubset(candidate_df.columns):
        bond = candidate_df[candidate_df["candidate_type"] == "bond"].copy()
        if not bond.empty:
            bond["is_true_bond_center"] = bond.groupby(["sample_id", "candidate_bond_local_idx"])["is_true_center"].transform("max")
            bond_unique = bond.drop_duplicates(subset=["sample_id", "candidate_bond_local_idx"])
            bond_unique = bond_unique.dropna(subset=["candidate_bde_percentile_in_mol"])
            if not bond_unique.empty:
                bins = np.linspace(0, 1, n_bins + 1)
                labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(n_bins)]
                bond_unique["bde_bin"] = pd.cut(bond_unique["candidate_bde_percentile_in_mol"], bins=bins, labels=labels, include_lowest=True)
                global_rate = float(bond_unique["is_true_bond_center"].mean())
                for b, g in bond_unique.groupby("bde_bin", observed=False):
                    rate = float(g["is_true_bond_center"].mean()) if len(g) else np.nan
                    rows.append({
                        "analysis_type": "bond_center_by_bond_bde",
                        "bde_bin": str(b),
                        "n_candidates": int(len(g)),
                        "true_rate": rate,
                        "global_true_rate": global_rate,
                        "enrichment": rate / global_rate if global_rate > 0 and np.isfinite(rate) else np.nan,
                    })

    # Atom-center enrichment: use atom min adjacent BDE percentile.
    if {"candidate_type", "candidate_min_adjacent_bde_percentile"}.issubset(candidate_df.columns):
        atom = candidate_df[candidate_df["candidate_type"] == "atom"].copy()
        atom = atom.dropna(subset=["candidate_min_adjacent_bde_percentile"])
        if not atom.empty:
            bins = np.linspace(0, 1, n_bins + 1)
            labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(n_bins)]
            atom["bde_bin"] = pd.cut(atom["candidate_min_adjacent_bde_percentile"], bins=bins, labels=labels, include_lowest=True)
            global_rate = float(atom["is_true_center"].mean())
            for b, g in atom.groupby("bde_bin", observed=False):
                rate = float(g["is_true_center"].mean()) if len(g) else np.nan
                rows.append({
                    "analysis_type": "atom_center_by_min_adjacent_bde",
                    "bde_bin": str(b),
                    "n_candidates": int(len(g)),
                    "true_rate": rate,
                    "global_true_rate": global_rate,
                    "enrichment": rate / global_rate if global_rate > 0 and np.isfinite(rate) else np.nan,
                })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def make_candidate_comparison(
    baseline_candidate_csv: str,
    bde_candidate_csv: str,
    outdir: str,
) -> Optional[pd.DataFrame]:
    """
    Merge baseline and +BDE candidate score/rank tables.
    This is the key table for rank-shift, BDE enrichment, and case studies.
    """
    if not baseline_candidate_csv or not bde_candidate_csv:
        return None
    if not os.path.exists(baseline_candidate_csv) or not os.path.exists(bde_candidate_csv):
        return None

    base = pd.read_csv(baseline_candidate_csv)
    bde = pd.read_csv(bde_candidate_csv)

    key_cols = [c for c in ["sample_id", "candidate_label"] if c in base.columns and c in bde.columns]
    if len(key_cols) < 2:
        key_cols = [c for c in ["sample_id", "candidate_type", "candidate_atom_i", "candidate_atom_j", "candidate_bond_change_type"] if c in base.columns and c in bde.columns]

    if len(key_cols) < 2:
        print("[Warning] Cannot merge candidate tables: no stable candidate keys.")
        return None

    merged = base.merge(bde, on=key_cols, how="inner", suffixes=("_baseline", "_bde"))

    if "candidate_rank_baseline" in merged.columns and "candidate_rank_bde" in merged.columns:
        merged["delta_candidate_rank"] = merged["candidate_rank_bde"] - merged["candidate_rank_baseline"]
    if "candidate_rank_score_baseline" in merged.columns and "candidate_rank_score_bde" in merged.columns:
        merged["delta_candidate_score"] = merged["candidate_rank_score_bde"] - merged["candidate_rank_score_baseline"]

    out_csv = os.path.join(outdir, "center_candidate_comparison.csv")
    merged.to_csv(out_csv, index=False)

    # BDE enrichment from +BDE candidate table.
    make_bde_enrichment_from_candidates(bde, os.path.join(outdir, "center_bde_enrichment.csv"))

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {os.path.join(outdir, 'center_bde_enrichment.csv')}")
    return merged



def write_manifest(outdir: str, model_tag: str, has_candidate: bool) -> None:
    rows = [
        {
            "csv_file": f"center_reaction_level_{model_tag}.csv",
            "status": "generated",
            "purpose": "Per-reaction top-k hit vector, best hit rank, skipped flag, product SMILES.",
            "used_for_panels": "b,f,g,h",
        },
        {
            "csv_file": f"center_result_long_{model_tag}.csv",
            "status": "generated",
            "purpose": "Long-format top-k hit table: one row per reaction and k.",
            "used_for_panels": "b,c",
        },
        {
            "csv_file": f"center_topk_summary_{model_tag}.csv",
            "status": "generated",
            "purpose": "Aggregated top-k accuracy for overall/bond-center/atom-center subsets.",
            "used_for_panels": "b,c",
        },
        {
            "csv_file": f"center_batch_progress_{model_tag}.csv",
            "status": "generated",
            "purpose": "Running accuracy over batches for debugging and training/test stability checks.",
            "used_for_panels": "supplementary",
        },
        {
            "csv_file": f"center_skipped_{model_tag}.csv",
            "status": "generated_if_any",
            "purpose": "Rows skipped by MolTreeFolder / preprocessing.",
            "used_for_panels": "quality control",
        },
        {
            "csv_file": f"center_candidate_table_{model_tag}.csv",
            "status": "generated" if has_candidate else "not_available_without_model_details",
            "purpose": "Candidate center score/rank table. Needed for BDE enrichment, class-stratified gain, molecular case studies.",
            "used_for_panels": "d,e,g",
        },
        {
            "csv_file": "center_reaction_comparison.csv",
            "status": "generated_in_compare_mode",
            "purpose": "Merged baseline vs +BDE per-reaction table with rank shift and corrected/worsened labels.",
            "used_for_panels": "f,g,h",
        },
        {
            "csv_file": "center_topk_comparison.csv",
            "status": "generated_in_compare_mode",
            "purpose": "Baseline vs +BDE top-k accuracy and delta.",
            "used_for_panels": "b,c",
        },
        {
            "csv_file": "center_case_type_summary.csv",
            "status": "generated_in_compare_mode",
            "purpose": "Counts of corrected, worsened, both_hit, both_miss samples.",
            "used_for_panels": "h",
        },
        {
            "csv_file": "center_rank_shift_summary.csv",
            "status": "generated_in_compare_mode",
            "purpose": "Summary statistics for true-center rank shift.",
            "used_for_panels": "f",
        },
    ]
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"center_csv_manifest_{model_tag}.csv"), index=False)


def compare_reaction_csvs(baseline_csv: str, bde_csv: str, outdir: str, knum: int = 10, baseline_candidate_csv: str = None, bde_candidate_csv: str = None) -> None:
    ensure_dir(outdir)

    base = pd.read_csv(baseline_csv)
    bde = pd.read_csv(bde_csv)

    # Prefer sample_id. Fallback to product_smiles if needed.
    key = "sample_id" if "sample_id" in base.columns and "sample_id" in bde.columns else "product_smiles"

    merged = base.merge(
        bde,
        on=key,
        how="inner",
        suffixes=("_baseline", "_bde"),
    )

    # Recover product_smiles in a clean column.
    if "product_smiles_baseline" in merged.columns:
        merged["product_smiles"] = merged["product_smiles_baseline"]
    elif "product_smiles" not in merged.columns and "product_smiles_bde" in merged.columns:
        merged["product_smiles"] = merged["product_smiles_bde"]

    if "best_hit_rank_baseline" in merged.columns and "best_hit_rank_bde" in merged.columns:
        merged["delta_rank"] = merged["best_hit_rank_bde"] - merged["best_hit_rank_baseline"]
        merged["rank_improved"] = (merged["delta_rank"] < 0).astype(int)
        merged["rank_worsened"] = (merged["delta_rank"] > 0).astype(int)

    for k in range(1, knum + 1):
        bcol = f"top{k}_hit_baseline"
        dcol = f"top{k}_hit_bde"
        if bcol in merged.columns and dcol in merged.columns:
            merged[f"delta_top{k}_hit"] = merged[dcol] - merged[bcol]

    def case_type(row, k=3):
        b = row.get(f"top{k}_hit_baseline", np.nan)
        d = row.get(f"top{k}_hit_bde", np.nan)
        if pd.isna(b) or pd.isna(d):
            return "unknown"
        b = int(float(b) >= 0.5)
        d = int(float(d) >= 0.5)
        if b == 0 and d == 1:
            return "corrected"
        if b == 1 and d == 0:
            return "worsened"
        if b == 1 and d == 1:
            return "both_hit"
        return "both_miss"

    for k in [1, 2, 3, 5, 10]:
        if k <= knum:
            merged[f"case_type_top{k}"] = merged.apply(lambda r, kk=k: case_type(r, kk), axis=1)

    merged.to_csv(os.path.join(outdir, "center_reaction_comparison.csv"), index=False)

    topk_rows = []
    for k in range(1, knum + 1):
        bcol = f"top{k}_hit_baseline"
        dcol = f"top{k}_hit_bde"
        if bcol not in merged.columns or dcol not in merged.columns:
            continue

        b_acc = float(pd.to_numeric(merged[bcol], errors="coerce").fillna(0).mean())
        d_acc = float(pd.to_numeric(merged[dcol], errors="coerce").fillna(0).mean())
        topk_rows.append(
            {
                "k": k,
                "baseline_accuracy": b_acc,
                "bde_accuracy": d_acc,
                "delta_accuracy": d_acc - b_acc,
                "n": int(len(merged)),
            }
        )
    pd.DataFrame(topk_rows).to_csv(os.path.join(outdir, "center_topk_comparison.csv"), index=False)

    case_rows = []
    for k in [1, 2, 3, 5, 10]:
        col = f"case_type_top{k}"
        if col not in merged.columns:
            continue
        vc = merged[col].value_counts(dropna=False)
        for case, count in vc.items():
            case_rows.append(
                {
                    "k": k,
                    "case_type": case,
                    "count": int(count),
                    "fraction": float(count / max(1, len(merged))),
                }
            )
    pd.DataFrame(case_rows).to_csv(os.path.join(outdir, "center_case_type_summary.csv"), index=False)

    rank_rows = []
    if "delta_rank" in merged.columns:
        valid = pd.to_numeric(merged["delta_rank"], errors="coerce").dropna()
        rank_rows = [
            {"metric": "n", "value": int(len(valid))},
            {"metric": "mean_delta_rank", "value": float(valid.mean()) if len(valid) else np.nan},
            {"metric": "median_delta_rank", "value": float(valid.median()) if len(valid) else np.nan},
            {"metric": "fraction_rank_improved", "value": float((valid < 0).mean()) if len(valid) else np.nan},
            {"metric": "fraction_rank_worsened", "value": float((valid > 0).mean()) if len(valid) else np.nan},
            {"metric": "fraction_rank_unchanged", "value": float((valid == 0).mean()) if len(valid) else np.nan},
        ]
    pd.DataFrame(rank_rows).to_csv(os.path.join(outdir, "center_rank_shift_summary.csv"), index=False)

    # Corrected/worsened case lists for panel g/h.
    case_col = "case_type_top3" if "case_type_top3" in merged.columns else None
    if case_col is not None:
        corrected = merged[merged[case_col] == "corrected"].copy()
        worsened = merged[merged[case_col] == "worsened"].copy()
        both_miss = merged[merged[case_col] == "both_miss"].copy()
        corrected.to_csv(os.path.join(outdir, "center_corrected_cases_top3.csv"), index=False)
        worsened.to_csv(os.path.join(outdir, "center_worsened_cases_top3.csv"), index=False)

        failure = pd.concat([worsened, both_miss], ignore_index=True)
        if len(failure) > 0:
            def _failure_mode(row):
                ct = row.get("true_center_type_bde", row.get("true_center_type_baseline", ""))
                pct = row.get("true_candidate_bde_percentile_in_mol_bde", row.get("true_candidate_bde_percentile_in_mol_baseline", np.nan))
                adj_pct = row.get("true_candidate_min_adjacent_bde_percentile_bde", row.get("true_candidate_min_adjacent_bde_percentile_baseline", np.nan))
                try:
                    pct_val = float(pct)
                except Exception:
                    pct_val = np.nan
                try:
                    adj_val = float(adj_pct)
                except Exception:
                    adj_val = np.nan
                if str(ct) == "atom":
                    return "atom_center_dominated"
                if np.isfinite(pct_val) and pct_val > 0.6:
                    return "true_center_not_low_BDE"
                if np.isfinite(adj_val) and adj_val > 0.6:
                    return "true_center_not_adjacent_low_BDE"
                return "other_or_context_controlled"
            failure["failure_mode"] = failure.apply(_failure_mode, axis=1)
            failure.to_csv(os.path.join(outdir, "center_failure_cases_top3.csv"), index=False)
            fm = failure["failure_mode"].value_counts().reset_index()
            fm.columns = ["failure_mode", "count"]
            fm["fraction"] = fm["count"] / max(1, len(failure))
            fm.to_csv(os.path.join(outdir, "center_failure_mode_summary.csv"), index=False)

    make_candidate_comparison(baseline_candidate_csv, bde_candidate_csv, outdir)

    print(f"[Saved] {os.path.join(outdir, 'center_reaction_comparison.csv')}")
    print(f"[Saved] {os.path.join(outdir, 'center_topk_comparison.csv')}")
    print(f"[Saved] {os.path.join(outdir, 'center_case_type_summary.csv')}")
    print(f"[Saved] {os.path.join(outdir, 'center_rank_shift_summary.csv')}")


def run_evaluation(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dir(args.save_dir)

    model_tag = args.model_tag
    if model_tag is None or str(model_tag).strip() == "":
        model_tag = "model"

    output_prefix = os.path.join(args.save_dir, args.output)
    if "-" in args.model_path:
        output_prefix += ("_" + args.model_path.split("-")[1])

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)
    avocab = common_atom_vocab

    model = MolCenter(vocab, common_atom_vocab, args)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    except Exception:
        raise ValueError("model does not exist")

    model = model.to(device)
    model.eval()

    data, meta_df = parse_test_file(args.test_path)
    meta_lookup = build_meta_lookup(meta_df)

    start = int(args.start)
    end = int(args.size) + start if args.size > 0 else len(data)
    data_eval = data[start:end]

    loader = MolTreeFolder(
        data_eval,
        vocab,
        avocab,
        args.batch_size,
        use_atomic=args.use_atomic,
        use_class=args.use_class,
        use_brics=args.use_brics,
        del_center=False,
        use_feature=args.use_feature,
    )

    knum = int(args.knum)
    topk_bool = np.zeros((len(loader.prod_list), knum), dtype=float)

    bond_bool_batches = []
    atom_bool_batches = []
    reaction_rows = []
    long_rows = []
    progress_rows = []
    skipped_rows = []
    candidate_tables = []

    # Keep original text outputs for compatibility.
    res_file = open(f"{output_prefix}_center_result.txt", "w")
    error_file = open(f"{output_prefix}_center_error.txt", "w")
    center_out = open(f"{output_prefix}_center.out", "w")

    time_start = time.time()
    num = 0

    for batch_idx, batch in enumerate(loader):
        classes, product_batch, product_tree, react_smiles, product_smiles, synthon_smiles, skip_idxs = batch

        batch_sample_ids = [str(idx) for idx, _ in product_smiles]
        batch_product_smiles = [str(smi) for _, smi in product_smiles]

        with torch.no_grad():
            top_acc, bond_center_acc, atom_center_acc, details = call_validate_centers(
                model,
                classes,
                product_batch,
                product_tree,
                knum=knum,
                sample_ids=batch_sample_ids,
                product_smiles_list=batch_product_smiles,
            )

        top_acc = np.asarray(top_acc, dtype=float)
        if top_acc.ndim == 1:
            top_acc = top_acc.reshape(1, -1)

        bond_center_acc = np.asarray(bond_center_acc, dtype=float)
        atom_center_acc = np.asarray(atom_center_acc, dtype=float)

        if bond_center_acc.size > 0:
            bond_bool_batches.append(bond_center_acc)
        if atom_center_acc.size > 0:
            atom_bool_batches.append(atom_center_acc)

        detail_df = normalize_candidate_details(details, model_tag=model_tag, batch_idx=batch_idx)
        batch_true_lookup = {}
        if not detail_df.empty:
            candidate_tables.append(detail_df)
            batch_true_lookup = make_true_center_lookup(detail_df)

        cur_acc = np.sum(top_acc[:, 0] == 1) / max(1, top_acc.shape[0])
        print("cur accuracy: %.4f" % cur_acc)

        acc_idx = 0
        batch_text = ""

        for i, (idx, product_smi) in enumerate(product_smiles):
            global_i = num + i
            sample_id = str(idx)
            skipped = int(i in skip_idxs)

            if skipped:
                hit_vec = np.zeros(knum, dtype=float)
                skipped_rows.append(
                    {
                        "model_tag": model_tag,
                        "batch_idx": batch_idx,
                        "sample_id": sample_id,
                        "product_smiles": product_smi,
                        "reason": "skip_idx_from_loader",
                    }
                )
            else:
                hit_vec = safe_float_array(top_acc[acc_idx, :], knum=knum)
                acc_idx += 1

            topk_bool[global_i, :] = hit_vec
            best_rank = first_hit_rank(hit_vec, miss_rank=knum + 1)

            meta = meta_lookup.get(sample_id, {})
            reactant_smi = meta.get("reactant_smiles", "")
            rxn_smi = meta.get("reaction_smiles", "")

            row = {
                "model_tag": model_tag,
                "global_index": global_i,
                "batch_idx": batch_idx,
                "sample_id": sample_id,
                "product_smiles": product_smi,
                "reactant_smiles": reactant_smi,
                "reaction_smiles": rxn_smi,
                "skipped": skipped,
                "best_hit_rank": best_rank,
                "hit_within_knum": int(best_rank <= knum),
            }

            if sample_id in batch_true_lookup:
                row.update(batch_true_lookup[sample_id])

            # Preserve extra metadata columns if available.
            for key, val in meta.items():
                if key not in row and key not in {"product_smiles_from_file"}:
                    row[key] = val

            for k in range(1, knum + 1):
                val = float(hit_vec[k - 1])
                row[f"top{k}_hit"] = val
                long_rows.append(
                    {
                        "model_tag": model_tag,
                        "sample_id": sample_id,
                        "global_index": global_i,
                        "k": k,
                        "hit": val,
                        "skipped": skipped,
                    }
                )

            reaction_rows.append(row)

            batch_text += "%s %s %s\n" % (
                sample_id,
                product_smi,
                " ".join([str(float(x)) for x in hit_vec]),
            )

        center_out.write(batch_text)
        center_out.flush()

        seen = num + len(product_smiles)
        topk_seen = np.sum(topk_bool[:seen, :], axis=0) / max(1, seen)
        num += len(product_smiles)

        if len(bond_bool_batches) > 0:
            bond_all = np.concatenate(bond_bool_batches, axis=0)
            bond_acc = np.nanmean(bond_all, axis=0)
            bond_n = bond_all.shape[0]
        else:
            bond_acc = np.full(knum, np.nan)
            bond_n = 0

        if len(atom_bool_batches) > 0:
            atom_all = np.concatenate(atom_bool_batches, axis=0)
            atom_acc = np.nanmean(atom_all, axis=0)
            atom_n = atom_all.shape[0]
        else:
            atom_acc = np.full(knum, np.nan)
            atom_n = 0

        print(
            "cur: top 5 accuracy: %s"
            % "  ".join([("%.4f" % x) for x in topk_seen[: min(5, knum)]])
        )
        print(
            "bond top 5[%d]: %s"
            % (bond_n, " ".join([("%.4f" % x) for x in bond_acc[: min(5, knum)]]))
        )
        print(
            "atom top 5[%d]: %s"
            % (atom_n, " ".join([("%.4f" % x) if np.isfinite(x) else "nan" for x in atom_acc[: min(5, knum)]]))
        )

        prog = {
            "model_tag": model_tag,
            "batch_idx": batch_idx,
            "seen_reactions": int(seen),
            "cur_batch_top1": float(cur_acc),
            "bond_center_n": int(bond_n),
            "atom_center_n": int(atom_n),
            "elapsed_sec": float(time.time() - time_start),
        }
        for k in range(1, knum + 1):
            prog[f"overall_top{k}"] = float(topk_seen[k - 1])
            prog[f"bond_top{k}"] = float(bond_acc[k - 1]) if k <= len(bond_acc) and np.isfinite(bond_acc[k - 1]) else np.nan
            prog[f"atom_top{k}"] = float(atom_acc[k - 1]) if k <= len(atom_acc) and np.isfinite(atom_acc[k - 1]) else np.nan

        progress_rows.append(prog)
        sys.stdout.flush()

    center_out.close()

    final_topk = np.sum(topk_bool, axis=0) / max(1, len(loader.prod_list))
    print(
        "top %d accuracy: %s"
        % (min(5, knum), "  ".join([("%.4f" % x) for x in final_topk[: min(5, knum)]]))
    )

    # Save original-style final text result.
    res_file.write(
        "top %d accuracy: %s\n"
        % (min(5, knum), "  ".join([("%.4f" % x) for x in final_topk[: min(5, knum)]]))
    )
    res_file.close()
    error_file.close()

    reaction_df = pd.DataFrame(reaction_rows)
    long_df = pd.DataFrame(long_rows)
    progress_df = pd.DataFrame(progress_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    reaction_csv = os.path.join(args.save_dir, f"center_reaction_level_{model_tag}.csv")
    long_csv = os.path.join(args.save_dir, f"center_result_long_{model_tag}.csv")
    progress_csv = os.path.join(args.save_dir, f"center_batch_progress_{model_tag}.csv")
    skipped_csv = os.path.join(args.save_dir, f"center_skipped_{model_tag}.csv")
    topk_csv = os.path.join(args.save_dir, f"center_topk_summary_{model_tag}.csv")

    reaction_df.to_csv(reaction_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    progress_df.to_csv(progress_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    topk_summary = make_topk_summary_from_reaction_rows(reaction_df, model_tag=model_tag, knum=knum)

    if len(bond_bool_batches) > 0:
        bond_all = np.concatenate(bond_bool_batches, axis=0)
    else:
        bond_all = None

    if len(atom_bool_batches) > 0:
        atom_all = np.concatenate(atom_bool_batches, axis=0)
    else:
        atom_all = None

    topk_summary = pd.concat(
        [
            topk_summary,
            make_subset_topk_summary(bond_all, model_tag, "bond_center", knum),
            make_subset_topk_summary(atom_all, model_tag, "atom_center", knum),
        ],
        ignore_index=True,
    )
    topk_summary.to_csv(topk_csv, index=False)

    has_candidate = False
    if len(candidate_tables) > 0:
        cand_df = pd.concat(candidate_tables, ignore_index=True)
        cand_csv = os.path.join(args.save_dir, f"center_candidate_table_{model_tag}.csv")
        cand_df.to_csv(cand_csv, index=False)
        make_bde_enrichment_from_candidates(cand_df, os.path.join(args.save_dir, f"center_bde_enrichment_{model_tag}.csv"))
        has_candidate = True
        print(f"[Saved] {cand_csv}")
        print(f"[Saved] {os.path.join(args.save_dir, f'center_bde_enrichment_{model_tag}.csv')}")
    else:
        print(
            "[Info] Candidate-level details were not returned by MolCenter.validate_centers. "
            "center_candidate_table is therefore not generated. To obtain BDE enrichment/case-study "
            "CSVs, expose candidate scores/ranks inside validate_centers."
        )

    write_manifest(args.save_dir, model_tag=model_tag, has_candidate=has_candidate)

    print(f"[Saved] {reaction_csv}")
    print(f"[Saved] {long_csv}")
    print(f"[Saved] {topk_csv}")
    print(f"[Saved] {progress_csv}")
    print(f"[Saved] {skipped_csv}")
    print(f"[Saved] {os.path.join(args.save_dir, f'center_csv_manifest_{model_tag}.csv')}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Evaluation mode: same arguments as original script.
    parser.add_argument("-t", "--test", dest="test_path")
    parser.add_argument("-m", "--model", dest="model_path")
    parser.add_argument("-d", "--save_dir", dest="save_dir", required=True)
    parser.add_argument("-o", "--output", dest="output", default="eval")
    parser.add_argument("-st", "--start", type=int, dest="start", default=0)
    parser.add_argument("-si", "--size", type=int, dest="size", default=0)

    parser.add_argument("--vocab", type=str, default="../data/vocab.txt")
    parser.add_argument("--knum", type=int, default=10)
    parser.add_argument("--ncpu", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--depthG", type=int, default=5)
    parser.add_argument("--depthT", type=int, default=3)

    parser.add_argument("--use_atomic", action="store_false")
    parser.add_argument("--sum_pool", action="store_false")
    parser.add_argument("--use_atom_product", action="store_true")
    parser.add_argument("--use_node_embed", action="store_true")
    parser.add_argument("--use_brics", action="store_true")
    parser.add_argument("--update_embed", action="store_true")
    parser.add_argument("--use_attachatom", action="store_true")
    parser.add_argument("--use_tree", action="store_true")
    parser.add_argument("--use_feature", action="store_false")
    parser.add_argument("--use_product", action="store_false")
    parser.add_argument("--use_class", action="store_true")
    parser.add_argument("--use_mess", action="store_true")
    parser.add_argument("--use_latent_attachatom", action="store_true")
    parser.add_argument("--network_type", type=str, default="gcn")

    # New analysis arguments.
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Name used in exported CSVs, e.g. baseline or bde.",
    )

    # Comparison mode: no model loading.
    parser.add_argument(
        "--compare_baseline_csv",
        type=str,
        default=None,
        help="Baseline center_reaction_level_*.csv for comparison mode.",
    )
    parser.add_argument(
        "--compare_bde_csv",
        type=str,
        default=None,
        help="+BDE center_reaction_level_*.csv for comparison mode.",
    )
    parser.add_argument(
        "--compare_baseline_candidate_csv",
        type=str,
        default=None,
        help="Optional baseline center_candidate_table_*.csv for candidate-level comparison.",
    )
    parser.add_argument(
        "--compare_bde_candidate_csv",
        type=str,
        default=None,
        help="Optional +BDE center_candidate_table_*.csv for candidate-level comparison and BDE enrichment.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.save_dir)

    if args.compare_baseline_csv and args.compare_bde_csv:
        compare_reaction_csvs(
            baseline_csv=args.compare_baseline_csv,
            bde_csv=args.compare_bde_csv,
            outdir=args.save_dir,
            knum=int(args.knum),
            baseline_candidate_csv=args.compare_baseline_candidate_csv,
            bde_candidate_csv=args.compare_bde_candidate_csv,
        )
        return

    if args.test_path is None or args.model_path is None:
        raise ValueError("Evaluation mode requires --test and --model. Comparison mode requires --compare_baseline_csv and --compare_bde_csv.")

    run_evaluation(args)


if __name__ == "__main__":
    main()

