"""Check shared routing and integrity of the benchmark candidate export."""
import csv
import argparse
import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_export", ROOT / "run_wetlab_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class EntryPointTests(unittest.TestCase):
    def test_worker_requires_strategy_and_rejects_mode(self):
        # Exercise the production parser without importing the neural runtime.
        core = ROOT / "BDGen" / "core"
        presets_tree = ast.parse((core / "bde_frag_gt" / "strategies.py").read_text())
        presets = next(ast.literal_eval(n.value) for n in presets_tree.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PRESETS" for t in n.targets))
        tree = ast.parse((core / "run_bde_frag_gt.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets))
        main.body = main.body[:end + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
        namespace = {"argparse": argparse, "PRESETS": presets}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])),
                     str(core / "run_bde_frag_gt.py"), "exec"), namespace)
        required = ["worker", "--smiles_file", "initial.smi"]
        with patch.object(sys, "argv", [*required, "--strategy", "dynamic_top3_regions"]):
            args = namespace["main"]()
        self.assertEqual(args.strategy, "dynamic_top3_regions")
        self.assertFalse(hasattr(args, "mode"))
        for suffix in ([], ["--strategy", "joint", "--mode", "global"]):
            with self.subTest(arguments=suffix), patch.object(sys, "argv", required + suffix), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    namespace["main"]()
                self.assertEqual(raised.exception.code, 2)

    def test_entry_points_use_the_same_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            options = ["--case", "MO-008", "--profile", "smoke", "--plan-only", "--output", directory]
            def plan(script, arguments):
                result = subprocess.run([sys.executable, str(ROOT / script), *arguments],
                                        check=True, capture_output=True, text=True)
                return json.loads(result.stdout)
            expected = plan("BDGen/run_bdgen.py", ["--strategy", "all", *options])
            self.assertEqual(plan("run_bde_frag_gt.py", ["--strategy", "all", *options]), expected)
            self.assertEqual(plan("run_wetlab_benchmark.py", ["--strategy", "all", *options]), expected)
            self.assertEqual(expected["jobs"], 6)
            for command in expected["commands"]:
                self.assertNotIn("--mode", command)
                self.assertIn("--strategy", command)
            self.assertFalse(Path(directory, "run_manifest.json").exists())

    def test_six_strategies_are_preserved_in_full_plan(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_wetlab_benchmark.py"), "--plan-only"],
                                check=True, capture_output=True, text=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["jobs"], 90)
        self.assertEqual(plan["cases"], [f"MO-{n:03d}" for n in range(1, 16)])
        self.assertEqual(plan["strategy_order"], list(benchmark.STRATEGIES))


class BenchmarkExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.folder = self.output / "MO-010" / "dynamic"
        self.folder.mkdir(parents=True)
        self.items = [dict(smiles="CCN", min_bde=81.0, qed=0.2),
                      dict(smiles="CCC", min_bde=88.0, qed=0.9)]
        self.candidates = self.folder / "candidates.json"
        self.candidates.write_text(json.dumps(self.items))
        self.original = self.candidates.read_bytes()
        self.job = dict(status="completed", case=dict(case_id="MO-010", initial_smiles="CCO"),
                        strategy="dynamic", budget=dict(number_molecules=3),
                        candidates_sha256=hashlib.sha256(self.original).hexdigest())
        self.save("job.json", self.job)
        self.save("candidate_selection.json", dict(requested=3, selected=2, shortfall=1))
        self.manifest = dict(status="completed", plan=dict(cases=["MO-010"],
                             strategy_order=["dynamic"], jobs=1),
                             results=[dict(case_id="MO-010", strategy="dynamic", status="completed")])
        (self.output / "run_manifest.json").write_text(json.dumps(self.manifest))
        self.pairs = self.output / "references.csv"
        with self.pairs.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "initial_smiles", "optimized_smiles", "target"])
            writer.writeheader()
            writer.writerow(dict(pair_id="source-11", initial_smiles="OCC", optimized_smiles="CCN", target="Example"))

    def save(self, name, value):
        (self.folder / name).write_text(json.dumps(value))

    def test_preserves_pareto_order_hashes_and_shortfall(self):
        benchmark.export_reports(self.output, self.pairs, top_k=1)
        report = self.output / "benchmark_reports"
        with (report / "wetlab_all_candidates.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["candidate_smiles"] for row in rows], ["CCN", "CCC"])
        self.assertEqual([row["selection_order"] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["case_id"], "MO-010")
        self.assertEqual(rows[0]["source_pair_id"], "source-11")
        with (report / "wetlab_generated_molecules_topk.csv").open(newline="") as handle:
            self.assertEqual([row["candidate_smiles"] for row in csv.DictReader(handle)], ["CCN"])
        with (report / "benchmark_summary.csv").open(newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["shortfall"], "1")
        self.assertEqual(self.candidates.read_bytes(), self.original)

    def test_rejects_modified_candidates_before_writing_report(self):
        self.candidates.write_text(json.dumps(list(reversed(self.items))))
        with self.assertRaisesRegex(ValueError, "Candidate hash differs"):
            benchmark.export_reports(self.output, self.pairs)
        self.assertFalse((self.output / "benchmark_reports").exists())

    def test_report_only_command_reuses_candidates(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "run_wetlab_benchmark.py"), "--report-only",
            "--output", str(self.output), "--pairs-csv", str(self.pairs),
            "--strategy", "dynamic", "--case", "MO-010", "--top-k", "1"],
            check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual((report["jobs"], report["candidates"]), (1, 2))
        self.assertEqual(self.candidates.read_bytes(), self.original)

    def test_rejects_incomplete_runs(self):
        self.manifest["status"] = "has_failed_jobs"
        (self.output / "run_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            benchmark.export_reports(self.output, self.pairs)

    def test_rejects_count_mismatch(self):
        self.save("candidate_selection.json", dict(requested=3, selected=3, shortfall=0))
        with self.assertRaisesRegex(ValueError, "Selection counts differ"):
            benchmark.export_reports(self.output, self.pairs)

    def test_rejects_duplicate_reference_structures(self):
        with self.pairs.open("a") as handle:
            handle.write("another,CCO,CCC,Example\n")
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            benchmark.export_reports(self.output, self.pairs)


if __name__ == "__main__":
    unittest.main()
