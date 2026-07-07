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

from awesom.create_dataset import SOM
from awesom.gpu_utils import print_device_info
from awesom.metrics_utils import ResultsLogger
from awesom.model import predict_ensemble

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'DataFrame.swapaxes' is deprecated and will be removed in a future version.*",
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def find_model_paths(checkpoints_path: str) -> List[str]:
    checkpoints_dir: Path = Path(checkpoints_path)
    model_paths: List[str] = []

    for model_dir in sorted(checkpoints_dir.glob("model_*")):
        checkpoint_path: Path = model_dir / "checkpoints" / "best_model.ckpt"
        if checkpoint_path.exists():
            model_paths.append(str(checkpoint_path))

    if not model_paths:
        direct_ckpts = sorted(checkpoints_dir.glob("*.ckpt"))
        model_paths = [str(p) for p in direct_ckpts]

    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_path}")

    return model_paths


def safe_metric(func, *args, default=np.nan, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


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

    for model_name, prob_col, model_path, is_ensemble in targets:
        y_prob = pred_df[prob_col].astype(float).values

        row = {
            "model_name": model_name,
            "model_path": model_path,
            "is_ensemble": int(is_ensemble),
            "prob_col": prob_col,
            "num_atoms": int(len(pred_df)),
            "num_molecules": int(pred_df["mol_id"].nunique()),
            "num_positive_atoms": int(pred_df["y_true"].sum()),
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


def print_first_batch_debug(dataloader: DataLoader) -> None:
    batch = next(iter(dataloader))

    print("[Debug] First batch")
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
        description="Test no-BDE ensemble model and export per-model metrics."
    )

    parser.add_argument("-i", "--input", required=True, help="Input data path")
    parser.add_argument("-c", "--checkpoints", required=True, help="Model checkpoints path")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["test", "infer"],
        required=True,
        help="Test or inference mode",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 means batch_size=len(data), same as original test.py.",
    )

    parser.add_argument(
        "--no_force_reprocess",
        action="store_true",
        help="Do not delete processed folder before loading dataset.",
    )

    parser.add_argument(
        "--debug_batch",
        action="store_true",
        help="Print first batch graph statistics.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print_device_info()

    labeled: bool = args.mode == "test"

    data: SOM = SOM(
        root=args.input,
        labeled=labeled,
        transform=T.ToUndirected(),
        force_reprocess=not args.no_force_reprocess,
    )

    print(f"Loaded {len(data)} instances for {args.mode}")

    model_paths: List[str] = find_model_paths(args.checkpoints)
    print(f"Found {len(model_paths)} model checkpoints")

    for p in model_paths:
        print(f"  - {p}")

    if args.batch_size <= 0:
        batch_size = len(data)
    else:
        batch_size = args.batch_size

    dataloader: DataLoader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.debug_batch:
        print_first_batch_debug(dataloader)

    predictions = predict_ensemble(dataloader, model_paths)
    predictions = predictions.to(torch.device("cpu"))

    print("Saving original ResultsLogger outputs...")

    if predictions:
        results_logger = ResultsLogger(args.output)
        results_logger.save_results(predictions, args.mode)

    print("Saving per-model prediction table...")

    pred_df = predictions.to_dataframe()
    pred_df = add_probability_ranks(pred_df)

    pred_path = os.path.join(args.output, "som_prediction_table.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    if args.mode == "test":
        model_names = predictions.model_names or [
            f"model_{i}" for i in range(predictions.logits.shape[0])
        ]
        model_paths_out = predictions.model_paths or model_paths

        metrics_df = build_metrics_summary(
            pred_df=pred_df,
            model_names=model_names,
            model_paths=model_paths_out,
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
    print(f"Saved {metrics_path}")

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
