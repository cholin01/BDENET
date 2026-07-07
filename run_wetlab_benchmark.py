from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bde_frag_gt.evaluate_properties import evaluate_smiles


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _canonicalize(smiles: str) -> str:
    """
    Convert SMILES to canonical isomeric SMILES.
    Reject empty SMILES and empty RDKit molecules.
    """
    smiles = "" if smiles is None else str(smiles).strip()

    if not smiles:
        raise ValueError("Empty SMILES.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if mol.GetNumAtoms() == 0:
        raise ValueError(f"Empty molecule from SMILES: {smiles}")

    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _similarity(left: str, right: str) -> float:
    """
    Morgan fingerprint Tanimoto similarity.
    """
    left_mol = Chem.MolFromSmiles(left)
    right_mol = Chem.MolFromSmiles(right)

    if left_mol is None or right_mol is None:
        return 0.0

    if left_mol.GetNumAtoms() == 0 or right_mol.GetNumAtoms() == 0:
        return 0.0

    left_fp = MORGAN_GENERATOR.GetFingerprint(left_mol)
    right_fp = MORGAN_GENERATOR.GetFingerprint(right_mol)

    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _safe_name(value: str) -> str:
    """
    Make a filesystem-safe directory name.
    """
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )


def _float_value(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    """
    Safely read float-like values from candidate dict.
    """
    value = row.get(key, default)
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _int_value(row: Dict[str, object], key: str, default: int = 0) -> int:
    """
    Safely read int-like values from candidate dict.
    """
    value = row.get(key, default)
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _druglike_rank_candidates(
    seed_smiles: str,
    candidates: List[Dict[str, object]],
    min_similarity: float,
    sa_max: float,
    logp_min: float,
    logp_max: float,
    mw_min: float,
    mw_max: float,
    tpsa_min: float,
    tpsa_max: float,
) -> List[Dict[str, object]]:
    """
    Rank candidates for outer benchmark top-k selection.

    Important:
    - BDE has already participated in the generation process through
      --maximize_bde and the downstream Pareto selection in run_bde_frag_gt.py.
    - This outer top-k ranking is drug-like-property oriented.
    - BDE values are retained in outputs but are not used as the primary
      outer top-k ranking criterion.
    """
    if not candidates:
        raise RuntimeError("BDGen produced no candidates.")

    ranked_candidates: List[Dict[str, object]] = []

    for row in candidates:
        row = dict(row)

        smiles = str(row.get("smiles", "")).strip()
        row["seed_similarity"] = _similarity(seed_smiles, smiles)

        ranked_candidates.append(row)

    def in_range(value: float, low: float, high: float) -> int:
        return 1 if low <= value <= high else 0

    def sort_key(row: Dict[str, object]) -> Tuple[int, int, int, int, int, int, float, float, float]:
        structural_ok = _int_value(row, "structural_filter_ok", 1)
        similarity = _float_value(row, "seed_similarity", 0.0)

        qed = _float_value(row, "qed", 0.0)
        sa = _float_value(row, "sa", 99.0)
        logp = _float_value(row, "logp", 999.0)
        mw = _float_value(row, "mw", 9999.0)
        tpsa = _float_value(row, "tpsa", 999.0)

        sim_ok = 1 if similarity >= min_similarity else 0
        sa_ok = 1 if sa <= sa_max else 0
        logp_ok = in_range(logp, logp_min, logp_max)
        mw_ok = in_range(mw, mw_min, mw_max)
        tpsa_ok = in_range(tpsa, tpsa_min, tpsa_max)

        property_range_score = logp_ok + mw_ok + tpsa_ok

        return (
            structural_ok,
            sim_ok,
            sa_ok,
            property_range_score,
            logp_ok,
            mw_ok,
            qed,
            similarity,
            -sa,
        )

    ranked_candidates = sorted(ranked_candidates, key=sort_key, reverse=True)

    for rank, row in enumerate(ranked_candidates, start=1):
        row["rank"] = rank

    return ranked_candidates


def _run_one(
    row_id: str,
    seed_smiles: str,
    mode: str,
    args: argparse.Namespace,
    population_file: Path,
) -> List[Dict[str, object]]:
    """
    Run one optimization job for one seed molecule and one mode.
    Return all ranked candidates generated by run_bde_frag_gt.py.
    """
    run_dir = Path(args.output_dir) / "runs" / _safe_name(row_id) / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).with_name("run_bde_frag_gt.py")),
        "--mode",
        mode,
        "--smiles_file",
        str(population_file),
        "--seed_smiles",
        seed_smiles,
        "--fragstore_path",
        args.fragstore_path,
        "--output_dir",
        str(run_dir),
        "--population_size",
        str(args.population_size),
        "--n_mutations",
        str(args.n_mutations),
        "--generations",
        str(args.generations),
        "--number_molecules",
        str(args.number_molecules),
        "--n_jobs",
        str(args.n_jobs),
        "--seed",
        str(args.seed),
        "--maximize_bde",
        "--keep_properties",
        "--min_similarity",
        str(args.min_similarity),
        "--structural_filter_profile",
        args.structural_filter_profile,
        "--sa_pass_threshold",
        str(args.sa_pass_threshold),

        # Physicochemical windows for both generation-time Pareto and final Pareto.
        "--logp_min",
        str(args.logp_min),
        "--logp_max",
        str(args.logp_max),
        "--mw_min",
        str(args.mw_min),
        "--mw_max",
        str(args.mw_max),
        "--tpsa_min",
        str(args.tpsa_min),
        "--tpsa_max",
        str(args.tpsa_max),
        "--sa_min",
        str(args.sa_min),
        "--sa_max",
        str(args.sa_max),

        # This enables generation-level Pareto selection in modified frag_gt.py.
        "--selection_strategy",
        args.selection_strategy,
    ]

    if args.bde_checkpoint:
        command.extend(
            [
                "--bde_checkpoint",
                args.bde_checkpoint,
                "--device",
                args.device,
            ]
        )
    elif not args.allow_heuristic:
        raise ValueError(
            "A BDENET checkpoint is required unless --allow_heuristic is explicitly set."
        )

    if args.admet_during_selection and args.admet_backend != "none":
        command.extend(
            [
                "--admet_backend",
                args.admet_backend,
                "--admet_endpoints",
                args.admet_endpoints,
                "--include_admet_in_pareto",
            ]
        )

        if args.admet_num_workers is not None:
            command.extend(["--admet_num_workers", str(args.admet_num_workers)])
    else:
        command.extend(["--admet_backend", args.admet_backend])

    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    summary_path = run_dir / "summary.json"
    all_candidates_path = run_dir / "summary_all_candidates.json"

    if all_candidates_path.exists():
        read_path = all_candidates_path
    elif summary_path.exists():
        read_path = summary_path
    else:
        raise FileNotFoundError(
            f"Neither summary_all_candidates.json nor summary.json was found in {run_dir}"
        )

    with open(read_path, encoding="utf-8") as handle:
        candidates = json.load(handle)

    ranked_candidates = _druglike_rank_candidates(
        seed_smiles=seed_smiles,
        candidates=candidates,
        min_similarity=args.min_similarity,
        sa_max=args.sa_max,
        logp_min=args.logp_min,
        logp_max=args.logp_max,
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        tpsa_min=args.tpsa_min,
        tpsa_max=args.tpsa_max,
    )

    # ------------------------------------------------------------------
    # Keep only args.number_molecules candidates for each seed and mode.
    #
    # FragGT may return population_size + n_mutations candidates, e.g.
    # 200 + 200 = 400. Here we explicitly keep only the top
    # args.number_molecules candidates after outer ranking.
    # ------------------------------------------------------------------
    retained_candidates = ranked_candidates[: args.number_molecules]

    # Re-assign rank after truncation.
    for rank, row in enumerate(retained_candidates, start=1):
        row["rank"] = rank
        row["mode"] = mode
        row["run_dir"] = str(run_dir)

    # Save retained candidates only.
    ranked_path = run_dir / "ranked_candidates.json"
    with open(ranked_path, "w", encoding="utf-8") as handle:
        json.dump(retained_candidates, handle, indent=2)

    # Overwrite summary.json so each seed/mode only has number_molecules molecules.
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(retained_candidates, handle, indent=2)

    # Overwrite summary_all_candidates.json as well, because you want to keep
    # only 100 molecules per seed/mode rather than all 400 intermediate candidates.
    summary_all_path = run_dir / "summary_all_candidates.json"
    with open(summary_all_path, "w", encoding="utf-8") as handle:
        json.dump(retained_candidates, handle, indent=2)

    # Keep optimized_smiles.json consistent with retained candidates.
    optimized_smiles_path = run_dir / "optimized_smiles.json"
    with open(optimized_smiles_path, "w", encoding="utf-8") as handle:
        json.dump(
            [row.get("smiles") for row in retained_candidates if row.get("smiles")],
            handle,
            indent=2,
        )

    return retained_candidates


def _write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: List[str] | None = None,
) -> None:
    """
    Write rows to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if not rows:
            raise ValueError(f"No rows to write and no fieldnames provided: {path}")
        fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_to_output_row(
    source: Dict[str, str],
    canonical_initial_smiles: str,
    generated: Dict[str, object],
) -> Dict[str, object]:
    """
    Convert one generated candidate to one output CSV row.
    """
    return {
        "pair_id": source.get("pair_id", ""),
        "initial_molecule": source.get("initial_molecule", ""),
        "initial_smiles": canonical_initial_smiles,
        "literature_optimized_molecule": source.get("optimized_molecule", ""),
        "literature_optimized_smiles": source.get("optimized_smiles", ""),
        "mode": generated.get("mode"),
        "rank": generated.get("rank"),
        "generated_smiles": generated.get("smiles"),
        "seed_similarity": generated.get("seed_similarity"),

        # BDE-related values.
        "min_bde": generated.get("min_bde"),
        "mean_bde": generated.get("mean_bde"),
        "weakest_bond_idx": generated.get("weakest_bond_idx"),
        "weakest_bond_type": generated.get("weakest_bond_type"),

        # Physicochemical properties.
        "qed": generated.get("qed"),
        "sa": generated.get("sa"),
        "logp": generated.get("logp"),
        "mw": generated.get("mw"),
        "tpsa": generated.get("tpsa"),
        "fraction_csp3": generated.get("fraction_csp3"),

        # Structural filters and alerts.
        "structural_filter_ok": generated.get("structural_filter_ok"),
        "structural_alert_count": generated.get("structural_alert_count"),

        # Pareto objectives from run_bde_frag_gt.py.
        "obj_min_bde": generated.get("obj_min_bde"),
        "obj_mean_bde": generated.get("obj_mean_bde"),
        "obj_qed": generated.get("obj_qed"),
        "obj_logp_window": generated.get("obj_logp_window"),
        "obj_mw_window": generated.get("obj_mw_window"),
        "obj_tpsa_window": generated.get("obj_tpsa_window"),
        "obj_sa_window": generated.get("obj_sa_window"),
        "obj_sa": generated.get("obj_sa"),
        "obj_structural_filter": generated.get("obj_structural_filter"),
        "obj_structural_alerts": generated.get("obj_structural_alerts"),

        # ADMET fields, may be None when --admet_backend none.
        "obj_admet_safety": generated.get("obj_admet_safety"),
        "obj_cyp_inhibition_safety": generated.get("obj_cyp_inhibition_safety"),
        "obj_admet_absorption": generated.get("obj_admet_absorption"),
        "admet_hERG": generated.get("admet_hERG"),
        "admet_AMES": generated.get("admet_AMES"),
        "admet_DILI": generated.get("admet_DILI"),
        "admet_ClinTox": generated.get("admet_ClinTox"),
        "admet_Clearance_Microsome_AZ": generated.get("admet_Clearance_Microsome_AZ"),
        "admet_Clearance_Hepatocyte_AZ": generated.get("admet_Clearance_Hepatocyte_AZ"),

        # Literature metadata.
        "doi": source.get("doi", ""),
        "source_url": source.get("source_url", ""),
        "run_dir": generated.get("run_dir"),
    }


def _flatten_rows(
    source_rows: List[Dict[str, str]],
    results: Dict[Tuple[str, str], List[Dict[str, object]]],
    top_k: int | None = None,
) -> List[Dict[str, object]]:
    """
    Flatten generated results to CSV rows.

    If top_k is None:
        keep all generated candidates.

    If top_k is an integer:
        keep only candidates with rank <= top_k for each seed and mode.
    """
    flattened: List[Dict[str, object]] = []
    available_modes = sorted({key[1] for key in results})

    for source in source_rows:
        canonical = _canonicalize(source["initial_smiles"])

        for mode in available_modes:
            generated_list = results.get((canonical, mode), [])

            if top_k is not None:
                generated_list = [
                    generated for generated in generated_list
                    if _int_value(generated, "rank", 10**9) <= top_k
                ]

            for generated in generated_list:
                flattened.append(
                    _candidate_to_output_row(
                        source=source,
                        canonical_initial_smiles=canonical,
                        generated=generated,
                    )
                )

    return flattened


def _evaluate_and_update_candidates(
    results: Dict[Tuple[str, str], List[Dict[str, object]]],
    canonical_to_rows: Dict[str, List[Dict[str, str]]],
    modes: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """
    Post-hoc evaluate all retained candidates and update candidate dictionaries.

    If --admet_backend none, evaluate_smiles should only compute basic properties.
    """
    distribution: Dict[str, object] = {
        "initial": {},
        "modes_all_candidates": {},
        "modes_topk_candidates": {},
    }

    initial_smiles = list(canonical_to_rows)

    if initial_smiles:
        initial_report = evaluate_smiles(
            initial_smiles,
            sa_pass_threshold=args.sa_pass_threshold,
            admet_backend=args.admet_backend,
            admet_endpoints=args.admet_endpoints,
            admet_num_workers=args.admet_num_workers,
        )
        distribution["initial"] = initial_report["summary"]

    for mode in modes:
        mode_items: List[Dict[str, object]] = []
        mode_smiles: List[str] = []

        for seed in canonical_to_rows:
            for item in results.get((seed, mode), []):
                smiles = str(item.get("smiles", "")).strip()
                if smiles:
                    mode_items.append(item)
                    mode_smiles.append(smiles)

        if not mode_smiles:
            distribution["modes_all_candidates"][mode] = {}
            distribution["modes_topk_candidates"][mode] = {}
            continue

        mode_report = evaluate_smiles(
            mode_smiles,
            sa_pass_threshold=args.sa_pass_threshold,
            admet_backend=args.admet_backend,
            admet_endpoints=args.admet_endpoints,
            admet_num_workers=args.admet_num_workers,
        )

        for item, property_row in zip(mode_items, mode_report["per_molecule"]):
            item.update(property_row)

        distribution["modes_all_candidates"][mode] = mode_report["summary"]

        topk_smiles: List[str] = []
        for seed in canonical_to_rows:
            topk_items = [
                item for item in results.get((seed, mode), [])
                if _int_value(item, "rank", 10**9) <= args.top_k
            ]
            for item in topk_items:
                smiles = str(item.get("smiles", "")).strip()
                if smiles:
                    topk_smiles.append(smiles)

        if topk_smiles:
            topk_report = evaluate_smiles(
                topk_smiles,
                sa_pass_threshold=args.sa_pass_threshold,
                admet_backend=args.admet_backend,
                admet_endpoints=args.admet_endpoints,
                admet_num_workers=args.admet_num_workers,
            )
            distribution["modes_topk_candidates"][mode] = topk_report["summary"]
        else:
            distribution["modes_topk_candidates"][mode] = {}

    return distribution


def _load_and_filter_source_rows(
    pairs_csv: str,
    output_dir: Path,
) -> tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    """
    Load input wet-lab pairs and skip invalid initial_smiles.
    """
    with open(pairs_csv, encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))

    if not raw_rows:
        raise ValueError("The wet-lab pair CSV is empty.")

    valid_rows: List[Dict[str, str]] = []
    skipped_rows: List[Dict[str, object]] = []
    canonical_to_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in raw_rows:
        pair_id = row.get("pair_id", "UNKNOWN")
        raw_smiles = row.get("initial_smiles", "")

        try:
            canonical = _canonicalize(raw_smiles)
        except Exception as exc:
            skipped_rows.append(
                {
                    "pair_id": pair_id,
                    "initial_molecule": row.get("initial_molecule", ""),
                    "initial_smiles": raw_smiles,
                    "reason": str(exc),
                }
            )
            print(
                f"[SKIP] {pair_id}: invalid initial_smiles={raw_smiles!r}; reason={exc}",
                flush=True,
            )
            continue

        valid_rows.append(row)
        canonical_to_rows[canonical].append(row)

    if skipped_rows:
        skipped_path = output_dir / "skipped_invalid_rows.csv"
        _write_csv(skipped_path, skipped_rows)
        print(f"[WARN] Skipped {len(skipped_rows)} invalid rows: {skipped_path}", flush=True)

    if not valid_rows:
        raise ValueError("No valid initial_smiles found after filtering.")

    return valid_rows, canonical_to_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark BDGen on literature wet-lab optimization pairs."
    )

    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fragstore_path", required=True)

    parser.add_argument("--bde_checkpoint", default=None)
    parser.add_argument("--allow_heuristic", action="store_true")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--modes", default="global,local")
    parser.add_argument("--population_size", type=int, default=80)
    parser.add_argument("--n_mutations", type=int, default=80)
    parser.add_argument("--generations", type=int, default=20)

    # How many candidates run_bde_frag_gt.py finally returns per seed/mode.
    parser.add_argument("--number_molecules", type=int, default=20)

    # How many candidates the outer benchmark keeps per seed/mode.
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Local-similarity and filters.
    parser.add_argument("--min_similarity", type=float, default=0.55)
    parser.add_argument("--structural_filter_profile", default="druglike_local")
    parser.add_argument("--sa_pass_threshold", type=float, default=6.0)

    # Physicochemical windows passed to downstream generation-time Pareto
    # and used by the outer top-k ranking.
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

    # This is passed to run_bde_frag_gt.py.
    # "pareto" requires your modified frag_gt/frag_gt.py.
    parser.add_argument("--selection_strategy", choices=["score", "pareto"], default="pareto")

    # ADMET is optional. Use --admet_backend none to avoid ADMET-AI dependency.
    parser.add_argument("--admet_backend", choices=["none", "admet_ai"], default="none")
    parser.add_argument("--admet_endpoints", default="recommended")
    parser.add_argument("--admet_num_workers", type=int, default=None)
    parser.add_argument(
        "--admet_during_selection",
        action="store_true",
        help=(
            "Load ADMET-AI inside every optimization run and include ADMET in downstream Pareto selection. "
            "Requires --admet_backend admet_ai."
        ),
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")

    if args.number_molecules <= 0:
        raise ValueError("--number_molecules must be positive.")

    if args.top_k > args.number_molecules:
        print(
            f"[WARN] --top_k {args.top_k} is larger than --number_molecules "
            f"{args.number_molecules}. Only available candidates will be kept.",
            flush=True,
        )

    if args.admet_during_selection and args.admet_backend == "none":
        raise ValueError("--admet_during_selection requires --admet_backend admet_ai.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows, canonical_to_rows = _load_and_filter_source_rows(
        pairs_csv=args.pairs_csv,
        output_dir=output_dir,
    )

    population_file = output_dir / "initial_population.smi"
    population_file.write_text(
        "\n".join(canonical_to_rows.keys()) + "\n",
        encoding="utf-8",
    )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if any(mode not in {"global", "local"} for mode in modes):
        raise ValueError("Supported benchmark modes are global and local.")

    with open(output_dir / "benchmark_run_params.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    results: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for index, seed_smiles in enumerate(canonical_to_rows, start=1):
        row_id = canonical_to_rows[seed_smiles][0].get("pair_id", f"seed_{index}")

        print(
            f"[{index}/{len(canonical_to_rows)}] {row_id}: {', '.join(modes)}",
            flush=True,
        )

        for mode in modes:
            ranked_candidates = _run_one(
                row_id=row_id,
                seed_smiles=seed_smiles,
                mode=mode,
                args=args,
                population_file=population_file,
            )

            results[(seed_smiles, mode)] = ranked_candidates

    distribution = _evaluate_and_update_candidates(
        results=results,
        canonical_to_rows=canonical_to_rows,
        modes=modes,
        args=args,
    )

    all_rows = _flatten_rows(
        source_rows=source_rows,
        results=results,
        top_k=None,
    )

    topk_rows = _flatten_rows(
        source_rows=source_rows,
        results=results,
        top_k=args.top_k,
    )

    _write_csv(output_dir / "wetlab_all_candidates.csv", all_rows)
    _write_csv(output_dir / "wetlab_generated_molecules_topk.csv", topk_rows)

    # Backward-compatible output name.
    _write_csv(output_dir / "wetlab_generated_molecules.csv", topk_rows)

    with open(output_dir / "property_distributions.json", "w", encoding="utf-8") as handle:
        json.dump(distribution, handle, indent=2)

    print("=" * 80, flush=True)
    print(f"Benchmark complete: {output_dir}", flush=True)
    print(f"All candidates saved to: {output_dir / 'wetlab_all_candidates.csv'}", flush=True)
    print(f"Top-{args.top_k} candidates saved to: {output_dir / 'wetlab_generated_molecules_topk.csv'}", flush=True)
    print(f"Backward-compatible top-k CSV: {output_dir / 'wetlab_generated_molecules.csv'}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()