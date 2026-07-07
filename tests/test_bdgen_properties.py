from rdkit import Chem

from bde_frag_gt.evaluate_properties import evaluate_smiles
from bde_frag_gt.optimizer import BDEOptimizationConfig, BDEPropertyScorer, HeuristicBDEPredictor
from bde_frag_gt.reaction_rules import structural_filter_report


def test_rdkit_property_panel_and_structural_alerts():
    report = evaluate_smiles(["CCO", "c1ccccc1N=Nc2ccccc2"])

    assert report["summary"]["n_valid"] == 2
    ethanol, azo = report["per_molecule"]
    assert ethanol["structural_alert_count"] == 0
    assert azo["structural_alert_count"] > 0
    assert "fraction_csp3" in ethanol
    assert "veber_pass" in ethanol


def test_druglike_local_filter_rejects_alerted_chemistry():
    predictor = HeuristicBDEPredictor()
    safe = structural_filter_report(Chem.MolFromSmiles("CCO"), predictor)
    alerted = structural_filter_report(Chem.MolFromSmiles("c1ccccc1N=Nc2ccccc2"), predictor)

    assert safe["ok"] is True
    assert alerted["ok"] is False
    assert alerted["reason"] == "structural_alert"


def test_scorer_normalizes_only_active_objectives():
    predictor = HeuristicBDEPredictor()
    config = BDEOptimizationConfig(mode="global")
    score = BDEPropertyScorer(predictor=predictor, config=config).score("CCO")

    assert 0.0 < score < 1.0
