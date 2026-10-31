#!/usr/bin/env python3
"""Run the BDGen benchmark and export candidates with literature-pair metadata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STRATEGIES = ("property-only", "bde-only", "joint", "random", "static", "dynamic")
PROPERTIES = ("min_bde", "mean_bde", "qed", "sa", "mw", "logp", "tpsa")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonicalize(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    require(mol is not None and mol.GetNumAtoms() > 0, f"Invalid molecule: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_references(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require({"initial_smiles", "optimized_smiles"} <= set(reader.fieldnames or []),
                "Reference CSV needs initial_smiles and optimized_smiles columns")
        rows = list(reader)
    require(bool(rows), "Reference CSV is empty")
    references = {}
    for row in rows:
        initial = canonicalize(row["initial_smiles"])
        require(initial not in references, "Ambiguous duplicate initial structure in reference CSV")
        references[initial] = dict(
            source_pair_id=row.get("pair_id", ""), initial_smiles=initial,
            literature_optimized_smiles=canonicalize(row["optimized_smiles"]),
            target=row.get("target", ""))
    return references


def command_for(args):
    command = [sys.executable, str(ROOT / "BDGen" / "run_bdgen.py"), "--strategy", args.strategy,
               "--output", str(args.output.resolve()), "--profile", args.profile,
               "--jobs", str(args.jobs)]
    for case_id in args.case_ids or []:
        command.extend(["--case", case_id])
    for flag, value in (("--cases", args.cases), ("--checkpoint", args.checkpoint),
                        ("--fragment-library", args.fragment_library),
                        ("--dependency-path", args.dependency_path)):
        if value is not None:
            command.extend([flag, str(value.expanduser().resolve())])
    if args.resume:
        command.append("--resume")
    if args.plan_only:
        command.append("--plan-only")
    return command


def validate_reference_cases(args):
    references = load_references(args.pairs_csv)
    cases = load(args.cases or ROOT / "BDGen/data/cases.json")
    selected = set(args.case_ids) if args.case_ids else {row["case_id"] for row in cases}
    require(selected <= {row["case_id"] for row in cases}, "Unknown --case ID")
    for row in cases:
        if row["case_id"] in selected:
            require(canonicalize(row["initial_smiles"]) in references,
                    f"No reference structure for {row['case_id']}")


def mean_value(items, key):
    values = [float(item[key]) for item in items if item.get(key) is not None]
    require(all(math.isfinite(value) for value in values), f"Nonfinite property: {key}")
    return statistics.mean(values) if values else None


def csv_text(rows, fields):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return handle.getvalue()


def write_atomic(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def export_reports(output, pairs_csv, case_ids=None, strategy="all", top_k=10):
    """Export verified final selections, preserving their recorded order."""
    output = Path(output).resolve()
    require(top_k > 0, "--top-k must be positive")
    manifest = load(output / "run_manifest.json")
    require(manifest["status"] == "completed", "Run is incomplete; inspect run_manifest.json")
    plan = manifest["plan"]
    require(all(re.fullmatch(r"[A-Za-z0-9_-]+", case_id) for case_id in plan["cases"]),
            "Invalid case ID in manifest")
    expected = {(case_id, name) for case_id in plan["cases"] for name in plan["strategy_order"]}
    results = manifest["results"]
    require(len(expected) == plan["jobs"] == len(results), "Run manifest job count differs")
    require({(row["case_id"], row["strategy"]) for row in results} == expected,
            "Run manifest has missing or duplicate jobs")
    require(all(row["status"] in {"completed", "reused_completed"} for row in results),
            "Run manifest contains failed jobs")
    require(set(plan["strategy_order"]) <= set(STRATEGIES), "Unknown strategy in manifest")
    selected_cases = set(case_ids) if case_ids else set(plan["cases"])
    require(selected_cases <= set(plan["cases"]), "Unknown case in report request")
    names = plan["strategy_order"] if strategy == "all" else [strategy]
    require(set(names) <= set(plan["strategy_order"]), "Strategy absent from this run")
    references = load_references(pairs_csv)
    all_rows, top_rows, summary_rows, reference_rows, evidence = [], [], [], [], []
    for case_id in plan["cases"]:
        if case_id not in selected_cases:
            continue
        initial = None
        for name in names:
            folder = output / case_id / name
            job = load(folder / "job.json")
            require(job["status"] == "completed" and job["case"]["case_id"] == case_id
                    and job["strategy"] == name, f"Unexpected job identity: {folder}")
            current_initial = canonicalize(job["case"]["initial_smiles"])
            if initial is None:
                initial = current_initial
                require(initial in references, f"No reference structure for {case_id}")
                reference_rows.append(dict(case_id=case_id, **references[initial]))
            require(current_initial == initial, f"Initial structures differ between strategies: {case_id}")
            candidate_path = folder / "candidates.json"
            data = candidate_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            require(sha == job["candidates_sha256"], f"Candidate hash differs: {folder}")
            items = json.loads(data)
            selection = load(folder / "candidate_selection.json")
            requested = job["budget"]["number_molecules"]
            require(selection["selected"] == len(items) <= requested
                    and selection["requested"] == requested
                    and selection["shortfall"] == requested - len(items),
                    f"Selection counts differ: {folder}")
            structures = [canonicalize(item["smiles"]) for item in items]
            require(len(structures) == len(set(structures)) and initial not in structures,
                    f"Duplicate or initial structure in final selection: {folder}")
            metadata = dict(case_id=case_id, strategy=name, **references[initial])
            for order, item in enumerate(items, 1):
                row = dict(metadata, selection_order=order,
                           **{"candidate_" + key: value for key, value in item.items()})
                all_rows.append(row)
                if order <= top_k:
                    top_rows.append(row)
            summary_rows.append(dict(case_id=case_id, strategy=name, requested=requested,
                                     selected=len(items), shortfall=requested - len(items),
                                     **{"mean_" + key: mean_value(items, key) for key in PROPERTIES}))
            evidence.append(dict(case_id=case_id, strategy=name, candidates_sha256=sha))
    fields = ["case_id", "strategy", "source_pair_id", "initial_smiles",
              "literature_optimized_smiles", "target", "selection_order"]
    fields += sorted({key for row in all_rows for key in row} - set(fields))
    summary_fields = ["case_id", "strategy", "requested", "selected", "shortfall"]
    summary_fields += ["mean_" + key for key in PROPERTIES]
    reference_fields = ["case_id", "source_pair_id", "initial_smiles", "literature_optimized_smiles", "target"]
    payloads = {
        "wetlab_all_candidates.csv": csv_text(all_rows, fields),
        "wetlab_generated_molecules_topk.csv": csv_text(top_rows, fields),
        "benchmark_summary.csv": csv_text(summary_rows, summary_fields),
        "literature_references.csv": csv_text(reference_rows, reference_fields),
        "benchmark_report.json": json.dumps(dict(
            cases=[row["case_id"] for row in reference_rows], strategies=names, top_k=top_k,
            candidate_count=len(all_rows), job_count=len(summary_rows),
            selection_order="Order in each verified candidates.json; top-k is its prefix",
            reference_matching="Canonical isomeric initial SMILES",
            reference_csv_sha256=hashlib.sha256(Path(pairs_csv).read_bytes()).hexdigest(),
            run_manifest_sha256=hashlib.sha256((output / "run_manifest.json").read_bytes()).hexdigest(),
            source_candidates=evidence), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    destination = output / "benchmark_reports"
    destination.mkdir(exist_ok=True)
    for filename, text in payloads.items():
        write_atomic(destination / filename, text)
    return dict(reports=str(destination), candidates=len(all_rows), jobs=len(summary_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("all", *STRATEGIES), default="all")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", "--output_dir", type=Path, default=Path("outputs/wetlab_benchmark"))
    parser.add_argument("--pairs-csv", "--pairs_csv", type=Path,
                        default=ROOT / "data/molecule_optimization_wetlab_pairs_15_unique.csv")
    parser.add_argument("--profile", choices=("manuscript", "smoke"), default="manuscript")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fragment-library", type=Path)
    parser.add_argument("--dependency-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Export this prefix of the recorded Pareto selection in a separate CSV")
    args = parser.parse_args()
    require(args.jobs > 0 and args.top_k > 0, "--jobs and --top-k must be positive")
    require(not (args.plan_only and args.report_only), "Choose --plan-only or --report-only")
    if args.dependency_path:
        sys.path.insert(0, str(args.dependency_path.expanduser().resolve()))
    if not args.report_only:
        if not args.plan_only:
            validate_reference_cases(args)
        subprocess.run(command_for(args), check=True)
    if not args.plan_only:
        print(json.dumps(export_reports(args.output, args.pairs_csv, args.case_ids,
                                       args.strategy, args.top_k), ensure_ascii=False))


if __name__ == "__main__":
    main()
