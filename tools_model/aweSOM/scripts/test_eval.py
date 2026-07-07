import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_dir)

from awesom.create_dataset_eval import SOM
from awesom.gpu_utils import print_device_info
from awesom.model_eval import predict_ensemble

try:
    from awesom.metrics_utils import ResultsLogger
except Exception:
    ResultsLogger = None


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def find_model_paths(checkpoints_path: str) -> List[str]:
    path = Path(checkpoints_path)

    if path.is_file() and path.name.endswith(".ckpt"):
        return [str(path)]

    model_paths: List[str] = []

    for model_dir in sorted(path.glob("model_*")):
        checkpoint_path = model_dir / "checkpoints" / "best_model.ckpt"
        if checkpoint_path.exists():
            model_paths.append(str(checkpoint_path))

    if not model_paths:
        direct_ckpts = sorted(path.glob("*.ckpt"))
        model_paths = [str(p) for p in direct_ckpts]

    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_path}")

    return model_paths


def safe_metric(func, *args, default=np.nan, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    if n == 0:
        return np.nan

    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]

        if i == 0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob > left) & (y_prob <= right)

        if mask.sum() == 0:
            continue

        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "atom_roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "atom_pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "atom_accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "atom_precision": safe_metric(
            precision_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "atom_recall": safe_metric(
            recall_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "atom_f1": safe_metric(
            f1_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "atom_mcc": safe_metric(matthews_corrcoef, y_true, y_pred),
        "atom_brier": safe_metric(brier_score_loss, y_true, y_prob),
        "atom_ece_10bin": expected_calibration_error(y_true, y_prob, n_bins=10),
    }


def molecule_topk_recall(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str = "y_true",
    k: int = 3,
) -> float:
    hits = []

    for _, g in df.groupby("mol_id"):
        true_atoms = set(g.loc[g[label_col] == 1, "atom_idx"].astype(int).tolist())

        if len(true_atoms) == 0:
            continue

        pred_atoms = set(
            g.sort_values(prob_col, ascending=False)
            .head(k)["atom_idx"]
            .astype(int)
            .tolist()
        )

        hits.append(len(true_atoms & pred_atoms) > 0)

    if len(hits) == 0:
        return np.nan

    return float(np.mean(hits))


def molecule_mean_best_true_rank(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str = "y_true",
) -> float:
    ranks = []

    for _, g in df.groupby("mol_id"):
        g = g.copy()
        g["_rank"] = g[prob_col].rank(method="first", ascending=False)

        true_ranks = g.loc[g[label_col] == 1, "_rank"].tolist()
        if len(true_ranks) > 0:
            ranks.append(min(true_ranks))

    if len(ranks) == 0:
        return np.nan

    return float(np.mean(ranks))


def add_probability_ranks(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()

    prob_cols = [c for c in pred_df.columns if c.endswith("_prob")]
    prob_cols.append("ensemble_prob_mean")
    prob_cols = sorted(set(prob_cols))

    for col in prob_cols:
        if col not in pred_df.columns:
            continue

        if col == "ensemble_prob_mean":
            rank_col = "ensemble_rank_in_mol"
        else:
            rank_col = col.replace("_prob", "_rank_in_mol")

        pred_df[rank_col] = (
            pred_df.groupby("mol_id")[col]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    return pred_df


def build_metrics_summary(
    pred_df: pd.DataFrame,
    model_names: List[str],
    model_paths: List[str],
) -> pd.DataFrame:
    rows = []

    targets = []

    for name, path in zip(model_names, model_paths):
        prob_col = f"{name}_prob"
        if prob_col in pred_df.columns:
            targets.append((name, prob_col, path, False))

    targets.append(("ensemble", "ensemble_prob_mean", "", True))

    y_true = pred_df["y_true"].astype(int).values

    for model_name, prob_col, path, is_ensemble in targets:
        y_prob = pred_df[prob_col].astype(float).values

        row = {
            "model_name": model_name,
            "model_path": path,
            "is_ensemble": int(is_ensemble),
            "num_atoms": int(len(pred_df)),
            "num_molecules": int(pred_df["mol_id"].nunique()),
            "num_positive_atoms": int(pred_df["y_true"].sum()),
            "prob_col": prob_col,
        }

        row.update(binary_metrics(y_true, y_prob))

        for k in [1, 2, 3, 5]:
            row[f"mol_top{k}_recall"] = molecule_topk_recall(
                pred_df,
                prob_col,
                k=k,
            )

        row["mol_mean_best_true_rank"] = molecule_mean_best_true_rank(
            pred_df,
            prob_col,
        )

        rows.append(row)

    return pd.DataFrame(rows)


def merge_prediction_with_atom_features(
    pred_df: pd.DataFrame,
    atom_df: pd.DataFrame,
) -> pd.DataFrame:
    if atom_df is None or atom_df.empty:
        print(
            "[Warning] atom_feature_table is empty. "
            "Skip merging atom-level BDE features."
        )
        return pred_df

    merge_cols = ["mol_id", "atom_idx"]

    missing = [c for c in merge_cols if c not in atom_df.columns]
    if missing:
        print(
            f"[Warning] atom_feature_table lacks columns {missing}. "
            "Skip merging atom-level BDE features."
        )
        return pred_df

    atom_keep = atom_df.copy()

    for c in ["description", "smiles", "is_som"]:
        if c in atom_keep.columns and c in pred_df.columns:
            atom_keep = atom_keep.drop(columns=[c])

    merged = pred_df.merge(atom_keep, on=merge_cols, how="left")

    return merged


def print_first_batch_debug(dataloader: DataLoader) -> None:
    batch = next(iter(dataloader))

    print("[Debug] First batch information")
    print("  num_graphs:", batch.num_graphs)
    print("  num_nodes:", batch.num_nodes)
    print("  num_edges:", batch.edge_index.size(1))
    print("  x shape:", tuple(batch.x.shape))
    print("  edge_attr shape:", tuple(batch.edge_attr.shape))
    print("  positive labels:", int(batch.y.sum().item()))

    if batch.edge_attr is not None:
        edge_attr = batch.edge_attr.float()
        print("  edge_attr mean:", edge_attr.mean(dim=0))
        print("  edge_attr min:", edge_attr.min(dim=0).values)
        print("  edge_attr max:", edge_attr.max(dim=0).values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SOM ensemble and export BDE-SOM analysis tables."
    )

    parser.add_argument("-i", "--input", required=True, help="Input data root")
    parser.add_argument("-c", "--checkpoints", required=True, help="Model checkpoints root")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["test", "infer"],
        required=True,
        help="test: labeled data with metrics; infer: unlabeled inference",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 means using batch_size=len(data), matching the original test.py.",
    )

    parser.add_argument("--force_reprocess", action="store_true")
    parser.add_argument(
        "--som_index_base",
        choices=["zero", "one", "auto"],
        default="zero",
    )
    parser.add_argument("--bde_hot_quantile", type=float, default=0.20)

    parser.add_argument("--mean_bde", type=float, default=94.62390701792717)
    parser.add_argument("--std_bde", type=float, default=11.503478622893386)
    parser.add_argument("--mean_bdfe", type=float, default=84.48772566704848)
    parser.add_argument("--std_bdfe", type=float, default=12.50514247866427)

    parser.add_argument(
        "--no_to_undirected",
        action="store_true",
        help=(
            "Disable T.ToUndirected(). By default it is enabled to match "
            "the original test.py behavior."
        ),
    )

    parser.add_argument(
        "--skip_legacy_results_logger",
        action="store_true",
        help="Skip original ResultsLogger output.",
    )

    parser.add_argument(
        "--debug_batch",
        action="store_true",
        help="Print first-batch graph statistics for consistency checking.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print_device_info()

    labeled = args.mode == "test"

    transform = None if args.no_to_undirected else T.ToUndirected()

    if transform is None:
        print("[Eval] Transform: None")
    else:
        print("[Eval] Transform: T.ToUndirected()")

    data = SOM(
        root=args.input,
        labeled=labeled,
        transform=transform,
        mean_bde=args.mean_bde,
        std_bde=args.std_bde,
        mean_bdfe=args.mean_bdfe,
        std_bdfe=args.std_bdfe,
        bde_hot_quantile=args.bde_hot_quantile,
        som_index_base=args.som_index_base,
        force_reprocess=args.force_reprocess,
        rebuild_if_metadata_missing=True,
    )

    print(f"[Eval] Loaded {len(data)} molecules for mode={args.mode}")

    atom_df, bond_df = data.export_metadata_tables(args.output)

    print(f"[Eval] Saved atom_feature_table.csv: {len(atom_df)} rows")
    print(f"[Eval] Saved bond_bde_table.csv: {len(bond_df)} rows")

    model_paths = find_model_paths(args.checkpoints)

    print(f"[Eval] Found {len(model_paths)} checkpoints")
    for p in model_paths:
        print(f"  - {p}")

    if args.batch_size <= 0:
        batch_size = len(data)
    else:
        batch_size = args.batch_size

    print(f"[Eval] DataLoader batch_size={batch_size}")

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.debug_batch:
        print_first_batch_debug(dataloader)

    predictions = predict_ensemble(dataloader, model_paths)
    predictions = predictions.to(torch.device("cpu"))

    if not args.skip_legacy_results_logger and ResultsLogger is not None:
        try:
            print("[Eval] Saving original ResultsLogger outputs...")
            legacy_logger = ResultsLogger(args.output)
            legacy_logger.save_results(predictions, args.mode)
        except Exception as e:
            print(f"[Warning] Original ResultsLogger failed: {e}")

    pred_df = predictions.to_dataframe()
    pred_df = add_probability_ranks(pred_df)
    pred_df = merge_prediction_with_atom_features(pred_df, atom_df)

    pred_path = os.path.join(args.output, "som_prediction_table.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[Eval] Saved som_prediction_table.csv: {len(pred_df)} rows")

    if labeled:
        metrics_df = build_metrics_summary(
            pred_df=pred_df,
            model_names=predictions.model_names,
            model_paths=predictions.model_paths,
        )
    else:
        metrics_df = pd.DataFrame(
            [
                {
                    "mode": "infer",
                    "note": "No metrics computed because labels are unavailable.",
                    "num_atoms": len(pred_df),
                    "num_molecules": pred_df["mol_id"].nunique(),
                }
            ]
        )

    metrics_path = os.path.join(args.output, "panel_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[Eval] Saved panel_metrics_summary.csv: {len(metrics_df)} rows")

    print(f"[Eval] All results saved to: {args.output}")


if __name__ == "__main__":
    main()
