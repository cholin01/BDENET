from __future__ import annotations

from typing import Dict, List, Optional, Set

from rdkit import Chem

from bde_frag_gt.evaluate_properties import structural_alerts
from bde_frag_gt.optimizer import BDEPredictor


RULE_PROFILES: Dict[str, Dict[str, object]] = {
    "none": {"allowed_pairs": None, "max_structural_alerts": None},
    "druglike_local": {
        "allowed_pairs": {"C-C", "C-N", "C-O", "C-S", "N-C", "O-C", "S-C"},
        "max_structural_alerts": 0,
    },
}


def _pair_label(a: Chem.Atom, b: Chem.Atom) -> str:
    return f"{a.GetSymbol()}-{b.GetSymbol()}"


def structural_filter_report(
    mol: Chem.Mol,
    predictor: BDEPredictor,
    profile_name: str = "druglike_local",
    hot_max_bde: Optional[float] = None,
    hot_top_k: int = 3,
) -> Dict[str, object]:
    profile = RULE_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown structural filter profile: {profile_name}")
    if profile_name == "none":
        return {"ok": True, "reason": "profile_disabled", **structural_alerts(mol)}

    alerts = structural_alerts(mol)
    max_alerts = profile["max_structural_alerts"]
    if max_alerts is not None and int(alerts["structural_alert_count"]) > int(max_alerts):
        return {"ok": False, "reason": "structural_alert", **alerts}

    allowed_pairs: Optional[Set[str]] = profile["allowed_pairs"]  # type: ignore[assignment]
    bdes = predictor.predict_bond_bde(mol)
    hot_non_ring: List[tuple[int, float]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        bond_idx = bond.GetIdx()
        bde = float(bdes[bond_idx]) if bond_idx < len(bdes) else 0.0
        if hot_max_bde is not None and bde > hot_max_bde:
            continue
        hot_non_ring.append((bond_idx, bde))

    hot_non_ring.sort(key=lambda item: item[1])
    hot_ids = [bond_idx for bond_idx, _ in hot_non_ring[: max(1, hot_top_k)]]
    if not hot_ids:
        return {"ok": True, "reason": "no_hot_non_ring_bond", **alerts}

    for bond_idx in hot_ids:
        bond = mol.GetBondWithIdx(bond_idx)
        pair = _pair_label(bond.GetBeginAtom(), bond.GetEndAtom())
        reverse_pair = _pair_label(bond.GetEndAtom(), bond.GetBeginAtom())
        if allowed_pairs is None or pair in allowed_pairs or reverse_pair in allowed_pairs:
            return {"ok": True, "reason": "supported_hot_bond", **alerts}

    return {"ok": False, "reason": "unsupported_hot_bond_type", **alerts}


def rule_ok_for_local_edit(
    mol: Chem.Mol,
    predictor: BDEPredictor,
    profile_name: str = "druglike_local",
    hot_max_bde: Optional[float] = None,
    hot_top_k: int = 3,
) -> bool:
    """Backward-compatible name for the structural post-generation filter."""
    return bool(
        structural_filter_report(
            mol,
            predictor=predictor,
            profile_name=profile_name,
            hot_max_bde=hot_max_bde,
            hot_top_k=hot_top_k,
        )["ok"]
    )
