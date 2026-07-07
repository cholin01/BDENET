from __future__ import annotations

from typing import Dict, List, Union

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from bde_frag_gt.optimizer import (
    BDEPredictor,
    BDEPropertyScorer,
    HeuristicBDEPredictor,
    weakest_bond_info,
)
from bde_frag_gt.reaction_rules import structural_filter_report


def summarize_bde_and_properties(
    smiles_list: List[str],
    predictor: BDEPredictor | None = None,
    structural_filter_profile: str = "none",
    local_hot_max_bde: float | None = None,
    hot_top_k: int = 3,
) -> List[Dict[str, Union[str, float, int]]]:
    """Build a compact report for top candidates."""
    predictor = predictor or HeuristicBDEPredictor()
    scorer = BDEPropertyScorer(predictor=predictor)
    rows: List[Dict[str, Union[str, float, int]]] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        bdes = predictor.predict_bond_bde(mol)
        weak = weakest_bond_info(smi, predictor=predictor)
        filter_report = structural_filter_report(
            mol,
            predictor=predictor,
            profile_name=structural_filter_profile,
            hot_max_bde=local_hot_max_bde,
            hot_top_k=hot_top_k,
        )
        rows.append(
            {
                "smiles": smi,
                "score": scorer.score(smi),
                "min_bde": min(bdes) if bdes else 0.0,
                "mean_bde": (sum(bdes) / len(bdes)) if bdes else 0.0,
                "weakest_bond_idx": weak["weakest_bond_idx"],
                "weakest_bond_type": weak["weakest_bond_type"],
                "qed": QED.qed(mol),
                "logp": Crippen.MolLogP(mol),
                "mw": Descriptors.MolWt(mol),
                "structural_filter_ok": int(bool(filter_report["ok"])),
                "structural_filter_reason": str(filter_report["reason"]),
                "structural_alert_count": int(filter_report["structural_alert_count"]),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows
