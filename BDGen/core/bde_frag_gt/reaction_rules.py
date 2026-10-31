from __future__ import annotations

from typing import Dict

from rdkit import Chem

from bde_frag_gt.evaluate_properties import structural_alerts
from bde_frag_gt.optimizer import BDEPredictor


RULE_PROFILES: Dict[str, Dict[str, object]] = {
    "none": {"allowed_pairs": None, "max_structural_alerts": None},
    "druglike_common": {"allowed_pairs": None,"max_structural_alerts":0},
}


def structural_filter_report(
    mol: Chem.Mol,
    predictor: BDEPredictor,
    profile_name: str = "druglike_common",
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

    return {'ok': True, 'reason': 'common_alert_check', **alerts}
