from __future__ import annotations

import argparse
import json
import logging
import os
import random

import numpy as np
from rdkit import Chem
import torch

from bde_frag_gt.optimizer import (
    BDEOptimizationConfig,
    BDEPropertyScorer,
    HeuristicBDEPredictor,
    bde_hot_bonds,
    molecule_properties,
    weakest_bond_info,
)
from bde_frag_gt.evaluate_properties import (
    add_admet_composite_objectives,
    evaluate_smiles,
    summarize_property_rows,
)
from bde_frag_gt.refine import summarize_bde_and_properties
from frag_gt.frag_gt import FragGTGenerator
from frag_gt.fragstore_scripts.generate_fragstore import FragmentStoreCreator

BDENET_IMPORT_ERROR: ImportError | None = None
try:
    from BDENET.bde_predictor import BDEPredictor as BDENETPredictor
except ImportError as exc:
    BDENET_IMPORT_ERROR = exc
    logging.warning("BDENET predictor is unavailable: %s", exc)
    BDENETPredictor = None


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class NeuralBDEPredictor:
    """
    Convert BDENET outputs to a list ordered by RDKit bond index.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        if BDENETPredictor is None:
            raise ImportError(f"Cannot initialize BDENET predictor: {BDENET_IMPORT_ERROR}")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[BDENET] Loading checkpoint {checkpoint_path} on {self.device}...")
        self.engine = BDENETPredictor(checkpoint_path=checkpoint_path, device=self.device)
        print("[BDENET] Initialization complete.")

    def predict_bond_bde(self, mol: Chem.Mol | str) -> list[float]:
        if isinstance(mol, str):
            mol_obj = Chem.MolFromSmiles(mol)
            if mol_obj is None:
                return []
        else:
            mol_obj = mol

        raw_predictions = self.engine.predict(mol_obj)

        if raw_predictions is None:
            return []

        if isinstance(raw_predictions, dict):
            if any(index not in raw_predictions for index in range(mol_obj.GetNumBonds())):
                return []
            return [float(raw_predictions[index]) for index in range(mol_obj.GetNumBonds())]

        values = [float(value) for value in raw_predictions]
        return values if len(values) == mol_obj.GetNumBonds() else []


def _resolve_from_project_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _ensure_fragstore(smiles_file: str, fragstore_path: str) -> str:
    if os.path.exists(fragstore_path):
        return fragstore_path

    output_dir = os.path.dirname(fragstore_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    creator = FragmentStoreCreator(frag_scheme="brics")
    creator.create_gene_table(smiles_file=smiles_file)
    creator.create_gene_type_table()
    creator.save_fragstore_to_disc(fragstore_path)

    return fragstore_path


def _dominates(a: dict, b: dict, keys: list[str]) -> bool:
    ge_all = all(float(a.get(key, 0.0)) >= float(b.get(key, 0.0)) for key in keys)
    gt_any = any(float(a.get(key, 0.0)) > float(b.get(key, 0.0)) for key in keys)
    return ge_all and gt_any


def _crowding_distance(front: list[int], items: list[dict], keys: list[str]) -> dict[int, float]:
    if not front:
        return {}

    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}

    for key in keys:
        sorted_front = sorted(front, key=lambda idx: float(items[idx].get(key, 0.0)))
        min_value = float(items[sorted_front[0]].get(key, 0.0))
        max_value = float(items[sorted_front[-1]].get(key, 0.0))

        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        denom = max(max_value - min_value, 1e-12)

        for pos in range(1, len(sorted_front) - 1):
            prev_value = float(items[sorted_front[pos - 1]].get(key, 0.0))
            next_value = float(items[sorted_front[pos + 1]].get(key, 0.0))
            distances[sorted_front[pos]] += (next_value - prev_value) / denom

    return distances


def select_by_pareto(
    items: list[dict],
    objective_keys: list[str],
    n_select: int,
) -> tuple[list[int], list[list[int]]]:
    if not items:
        return [], []

    dominates = [set() for _ in items]
    dominated_count = [0 for _ in items]
    fronts: list[list[int]] = [[]]

    for i in range(len(items)):
        for j in range(len(items)):
            if i == j:
                continue

            if _dominates(items[i], items[j], objective_keys):
                dominates[i].add(j)
            elif _dominates(items[j], items[i], objective_keys):
                dominated_count[i] += 1

        if dominated_count[i] == 0:
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        next_front: list[int] = []

        for i in fronts[f]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)

        f += 1

    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            remaining = n_select - len(selected)
            distances = _crowding_distance(front, items, objective_keys)
            front_sorted = sorted(front, key=lambda idx: distances.get(idx, 0.0), reverse=True)
            selected.extend(front_sorted[:remaining])
            break

    return selected, fronts


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BDE-guided fragment evolutionary optimization.")

    parser.add_argument("--bde_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mode", type=str, choices=["global", "local"], default="global")
    parser.add_argument("--smiles_file", type=str, required=True)
    parser.add_argument("--seed_smiles", type=str, default=None)
    parser.add_argument(
        "--fragstore_path",
        type=str,
        default="frag_gt/data/fragment_libraries/chembl_33_chemreps_std_fragstore_brics_filter2.pkl",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/bde_frag_gt")
    parser.add_argument("--population_size", type=int, default=200)
    parser.add_argument("--n_mutations", type=int, default=200)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--number_molecules", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--target_min_bde", type=float, default=90.0)
    parser.add_argument("--target_mean_bde", type=float, default=105.0)
    parser.add_argument("--weakest_bde_low", type=float, default=None)
    parser.add_argument("--weakest_bde_high", type=float, default=None)
    parser.add_argument("--target_bond_idx", type=int, default=None)
    parser.add_argument("--target_bond_low", type=float, default=None)
    parser.add_argument("--target_bond_high", type=float, default=None)
    parser.add_argument("--maximize_target_bond", action="store_true")
    parser.add_argument("--maximize_bde", action="store_true")

    parser.add_argument("--keep_properties", action="store_true")
    parser.add_argument("--min_similarity", type=float, default=0.55)
    parser.add_argument("--hot_top_k", type=int, default=3)
    parser.add_argument("--hot_max_bde", type=float, default=None)

    parser.add_argument(
        "--structural_filter_profile",
        choices=["none", "druglike_local"],
        default="druglike_local",
    )

    parser.add_argument("--logp_min", type=float, default=0.0)
    parser.add_argument("--logp_max", type=float, default=4.0)
    parser.add_argument("--mw_min", type=float, default=150.0)
    parser.add_argument("--mw_max", type=float, default=550.0)
    parser.add_argument("--tpsa_min", type=float, default=0.0)
    parser.add_argument("--tpsa_max", type=float, default=140.0)
    parser.add_argument("--sa_min", type=float, default=1.0)
    parser.add_argument("--sa_max", type=float, default=6.0)
    parser.add_argument("--use_two_stage_bde", action="store_true")
    parser.add_argument("--bde_top_k_per_batch", type=int, default=40)
    parser.add_argument("--bde_top_fraction_per_batch", type=float, default=0.25)
    parser.add_argument("--bde_min_cheap_score", type=float, default=0.25)
    parser.add_argument("--max_bde_mw", type=float, default=650.0)
    parser.add_argument("--max_bde_heavy_atoms", type=int, default=70)
    parser.add_argument("--max_bde_bonds", type=int, default=75)

    parser.add_argument("--selection_strategy", choices=["score", "pareto"], default="pareto")

    parser.add_argument("--skip_auto_evaluate", action="store_true")
    parser.add_argument("--sa_pass_threshold", type=float, default=6.0)

    parser.add_argument(
        "--admet_backend",
        choices=["none", "admet_ai"],
        default="none",
    )
    parser.add_argument("--admet_endpoints", type=str, default="recommended")
    parser.add_argument("--admet_num_workers", type=int, default=None)
    parser.add_argument("--include_admet_in_pareto", action="store_true")

    args = parser.parse_args()

    args.smiles_file = _resolve_from_project_root(args.smiles_file)
    args.fragstore_path = _resolve_from_project_root(args.fragstore_path)
    args.output_dir = _resolve_from_project_root(args.output_dir)

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    weakest_range = None
    if args.weakest_bde_low is not None and args.weakest_bde_high is not None:
        weakest_range = (args.weakest_bde_low, args.weakest_bde_high)

    target_bond_range = None
    if args.target_bond_low is not None and args.target_bond_high is not None:
        target_bond_range = (args.target_bond_low, args.target_bond_high)

    config = BDEOptimizationConfig(
        mode=args.mode,
        target_min_bde=args.target_min_bde,
        target_mean_bde=args.target_mean_bde,
        weakest_bde_range=weakest_range,
        maximize_bde=args.maximize_bde,
        target_bond_idx=args.target_bond_idx,
        target_bond_range=target_bond_range,
        maximize_target_bond=args.maximize_target_bond,
        min_similarity=args.min_similarity,
        seed_smiles=args.seed_smiles if args.mode == "local" else None,
        structural_filter_profile=args.structural_filter_profile,
        local_hot_max_bde=args.hot_max_bde,
        hot_top_k=args.hot_top_k,
        logp_range=(args.logp_min, args.logp_max),
        mw_range=(args.mw_min, args.mw_max),
        tpsa_range=(args.tpsa_min, args.tpsa_max),
        sa_range=(args.sa_min, args.sa_max),
        max_tpsa=args.tpsa_max,
        max_sa=args.sa_max,
    )

    if args.bde_checkpoint:
        predictor = NeuralBDEPredictor(checkpoint_path=args.bde_checkpoint, device=args.device)
    else:
        print("[BDENET] Using HeuristicBDEPredictor for a lightweight baseline run.")
        predictor = HeuristicBDEPredictor()

    if args.mode == "local":
        if not args.seed_smiles:
            raise ValueError("--seed_smiles is required when --mode local")

        if args.keep_properties:
            base = molecule_properties(args.seed_smiles, predictor=predictor)
            config.min_qed = max(0.0, base["qed"] - 0.02)
            config.min_logp = base["logp"] - 0.2
            config.min_mw = max(0.0, base["mw"] - 10.0)
            config.max_tpsa = min(args.tpsa_max, base.get("tpsa", args.tpsa_max) + 15.0)
            config.max_sa = min(args.sa_max, base.get("sa", args.sa_max) + 0.5)

    elif args.keep_properties and args.seed_smiles:
        base = molecule_properties(args.seed_smiles, predictor=predictor)
        config.min_qed = max(0.0, base["qed"] - 0.02)
        config.min_logp = base["logp"] - 0.2
        config.min_mw = max(0.0, base["mw"] - 10.0)
        config.max_tpsa = min(args.tpsa_max, base.get("tpsa", args.tpsa_max) + 15.0)
        config.max_sa = min(args.sa_max, base.get("sa", args.sa_max) + 0.5)

    scorer = BDEPropertyScorer(predictor=predictor, config=config)

    fragstore_path = _ensure_fragstore(args.smiles_file, args.fragstore_path)

    n_jobs = args.n_jobs
    if args.bde_checkpoint and ("cuda" in args.device):
        print("Warning: Running neural network inference with PyTorch multiprocessing might cause CUDA initialization errors.")
        print("If it crashes, try running with --n_jobs 1.")

    generator = FragGTGenerator(
        smi_file=args.smiles_file,
        fragmentation_scheme="brics",
        fragstore_path=fragstore_path,
        allow_unspecified_stereo=True,
        scorer="counts",
        population_size=args.population_size,
        n_mutations=args.n_mutations,
        generations=args.generations,
        random_start=False,
        patience=8,
        n_jobs=n_jobs,
        selection_strategy=args.selection_strategy,
        pareto_objective_keys=scorer.pareto_objective_keys,
    )

    starting_population = None
    if args.seed_smiles:
        starting_population = [args.seed_smiles] * max(20, args.population_size)

    candidate_pool_size = max(args.number_molecules * 4, args.population_size)

    optimized = generator.optimize(
        scoring_function=scorer,
        number_molecules=candidate_pool_size,
        starting_population=starting_population,
        fixed_substructure_smarts=None,
        job_name="bde_frag_gt",
    )

    summary = summarize_bde_and_properties(
        optimized,
        predictor=predictor,
        structural_filter_profile=args.structural_filter_profile,
        local_hot_max_bde=args.hot_max_bde,
        hot_top_k=args.hot_top_k,
    )

    property_report = evaluate_smiles(
        [str(row["smiles"]) for row in summary],
        sa_pass_threshold=args.sa_pass_threshold,
        admet_backend=args.admet_backend,
        admet_endpoints=args.admet_endpoints,
        admet_num_workers=args.admet_num_workers,
    )

    properties_by_smiles = {
        str(row["smiles"]): row
        for row in property_report["per_molecule"]
        if int(row.get("valid", 0)) == 1
    }

    for row in summary:
        row.update(properties_by_smiles.get(str(row["smiles"]), {}))

        row["obj_min_bde"] = float(row["min_bde"])
        row["obj_mean_bde"] = float(row["mean_bde"])
        row["obj_qed"] = _float(row, "qed", 0.0)

        logp = _float(row, "logp", 999.0)
        mw = _float(row, "mw", 9999.0)
        tpsa = _float(row, "tpsa", 999.0)
        sa = _float(row, "sa", 99.0)

        row["obj_logp_window"] = 1.0 if args.logp_min <= logp <= args.logp_max else 0.0
        row["obj_mw_window"] = 1.0 if args.mw_min <= mw <= args.mw_max else 0.0
        row["obj_tpsa_window"] = 1.0 if args.tpsa_min <= tpsa <= args.tpsa_max else 0.0
        row["obj_sa_window"] = 1.0 if args.sa_min <= sa <= args.sa_max else 0.0
        row["obj_sa"] = 1.0 / (1.0 + max(0.0, sa))

        row["obj_structural_filter"] = float(row.get("structural_filter_ok", 1))
        row["obj_structural_alerts"] = 1.0 / (1.0 + float(row.get("structural_alert_count", 0)))

        add_admet_composite_objectives(row)

    objective_keys = [
        "obj_min_bde",
        "obj_mean_bde",
        "obj_qed",
        "obj_logp_window",
        "obj_mw_window",
        "obj_tpsa_window",
        "obj_sa_window",
        "obj_sa",
        "obj_structural_filter",
        "obj_structural_alerts",
    ]

    if args.include_admet_in_pareto:
        if args.admet_backend == "none":
            raise ValueError("--include_admet_in_pareto requires --admet_backend admet_ai")

        for key in ("obj_admet_safety", "obj_cyp_inhibition_safety", "obj_admet_absorption"):
            if all(key in row for row in summary):
                objective_keys.append(key)

    selected_idx, fronts = select_by_pareto(
        summary,
        objective_keys=objective_keys,
        n_select=args.number_molecules,
    )

    selected = [summary[i] for i in selected_idx]
    optimized = [str(x["smiles"]) for x in selected]

    with open(os.path.join(args.output_dir, "optimized_smiles.json"), "w", encoding="utf-8") as handle:
        json.dump(optimized, handle, indent=2)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    with open(os.path.join(args.output_dir, "summary_all_candidates.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(os.path.join(args.output_dir, "pareto_fronts.json"), "w", encoding="utf-8") as handle:
        json.dump([[summary[i] for i in front] for front in fronts], handle, indent=2)

    with open(os.path.join(args.output_dir, "run_params.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    if not args.skip_auto_evaluate:
        selected_property_rows = [
            properties_by_smiles[smi] for smi in optimized if smi in properties_by_smiles
        ]
        evaluation = {
            "backend": property_report["backend"],
            "summary": summarize_property_rows(selected_property_rows, args.sa_pass_threshold),
            "per_molecule": selected_property_rows,
        }

        with open(os.path.join(args.output_dir, "property_evaluation.json"), "w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2)

    if args.mode == "local":
        base_weak = weakest_bond_info(args.seed_smiles, predictor=predictor)
        with open(os.path.join(args.output_dir, "seed_weakest_bond.json"), "w", encoding="utf-8") as handle:
            json.dump(base_weak, handle, indent=2)

        hot = bde_hot_bonds(
            args.seed_smiles,
            predictor=predictor,
            top_k=args.hot_top_k,
            max_bde=args.hot_max_bde,
        )
        with open(os.path.join(args.output_dir, "seed_bde_hot_bonds.json"), "w", encoding="utf-8") as handle:
            json.dump(hot, handle, indent=2)

    print(f"Generated {len(optimized)} molecules.")
    print(f"Results written to: {args.output_dir}")
    print(f"Optimization selection strategy: {args.selection_strategy}")
    print(f"Final Pareto objectives: {objective_keys}")


if __name__ == "__main__":
    main()