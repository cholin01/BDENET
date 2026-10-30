from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors

try:
    from rdkit.Contrib.SA_Score import sascorer  # type: ignore
except Exception:
    sascorer = None


RECOMMENDED_ADMET_ENDPOINTS = (
    "HIA_Hou",
    "Bioavailability_Ma",
    "Solubility_AqSolDB",
    "Caco2_Wang",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "PPBR_AZ",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "hERG",
    "ClinTox",
    "AMES",
    "DILI",
    "LD50_Zhu",
)

ADMET_SAFETY_ENDPOINTS = ("hERG", "ClinTox", "AMES", "DILI")
ADMET_CYP_INHIBITION_ENDPOINTS = (
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
)
ADMET_ABSORPTION_ENDPOINTS = ("HIA_Hou", "Bioavailability_Ma", "PAMPA_NCATS")


def _sa_score(mol: Chem.Mol) -> float:
    if sascorer is not None:
        return float(sascorer.calculateScore(mol))

    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    score = 1.0 + 0.015 * heavy + 0.25 * rings + 0.20 * stereo + (0.8 - sp3)
    return float(max(1.0, min(10.0, score)))


@lru_cache(maxsize=1)
def _filter_catalogs() -> Mapping[str, FilterCatalog.FilterCatalog]:
    catalogs: Dict[str, FilterCatalog.FilterCatalog] = {}
    catalog_enum = FilterCatalog.FilterCatalogParams.FilterCatalogs
    for name, flag in (
        ("pains", catalog_enum.PAINS),
        ("brenk", catalog_enum.BRENK),
        ("nih", catalog_enum.NIH),
        ("zinc", catalog_enum.ZINC),
    ):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(flag)
        catalogs[name] = FilterCatalog.FilterCatalog(params)
    return catalogs


def structural_alerts(mol: Chem.Mol) -> Dict[str, object]:
    result: Dict[str, object] = {}
    all_names: List[str] = []
    for catalog_name, catalog in _filter_catalogs().items():
        names = sorted({entry.GetDescription() for entry in catalog.GetMatches(mol)})
        result[f"{catalog_name}_alert_count"] = len(names)
        result[f"{catalog_name}_alerts"] = names
        all_names.extend(f"{catalog_name}:{name}" for name in names)
    result["structural_alert_count"] = len(all_names)
    result["structural_alerts"] = all_names
    return result


def _calc_rdkit_props(smiles: str) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "valid": 0}

    row: Dict[str, object] = {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "valid": 1,
        "qed": float(QED.qed(mol)),
        "sa": _sa_score(mol),
        "logp": float(Crippen.MolLogP(mol)),
        "molar_refractivity": float(Crippen.MolMR(mol)),
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "hetero_atom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "stereo_center_count": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "bertz_complexity": float(Descriptors.BertzCT(mol)),
    }
    row["lipinski_pass"] = int(
        float(row["mw"]) <= 500
        and float(row["logp"]) <= 5
        and int(row["hbd"]) <= 5
        and int(row["hba"]) <= 10
    )
    row["veber_pass"] = int(float(row["tpsa"]) <= 140 and int(row["rot_bonds"]) <= 10)
    row.update(structural_alerts(mol))
    return row


def _resolve_admet_endpoints(endpoint_spec: str | Sequence[str] | None) -> List[str] | None:
    if endpoint_spec is None or endpoint_spec == "all":
        return None
    if endpoint_spec == "recommended":
        return list(RECOMMENDED_ADMET_ENDPOINTS)
    if isinstance(endpoint_spec, str):
        return [item.strip() for item in endpoint_spec.split(",") if item.strip()]
    return [str(item) for item in endpoint_spec]


def _predict_admet_ai(
    smiles_list: Sequence[str],
    endpoint_spec: str | Sequence[str] | None,
    num_workers: int | None,
) -> Dict[str, Dict[str, float]]:
    try:
        from admet_ai import ADMETModel
    except ImportError as exc:
        raise RuntimeError(
            "ADMET-AI is not installed. Install the optional backend with `pip install admet-ai`."
        ) from exc

    endpoints = _resolve_admet_endpoints(endpoint_spec)
    model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=num_workers)
    predictions = model.predict(list(smiles_list))
    if not hasattr(predictions, "iterrows"):
        raise RuntimeError("ADMET-AI returned an unexpected prediction object.")

    available = set(str(column) for column in predictions.columns)
    selected = sorted(available) if endpoints is None else [name for name in endpoints if name in available]
    missing = [] if endpoints is None else [name for name in endpoints if name not in available]
    if missing:
        logging.warning("ADMET-AI endpoints not available and skipped: %s", ", ".join(missing))

    result: Dict[str, Dict[str, float]] = {}
    for index, values in predictions.iterrows():
        result[str(index)] = {
            f"admet_{name}": float(values[name])
            for name in selected
            if values[name] is not None and math.isfinite(float(values[name]))
        }
    return result


def _numeric_mean(rows: Iterable[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
    return float(mean(values)) if values else None


def summarize_property_rows(rows: Sequence[Mapping[str, object]], sa_pass_threshold: float) -> Dict[str, object]:
    valid_rows = [row for row in rows if int(row.get("valid", 0)) == 1]
    summary: Dict[str, object] = {
        "n_total": len(rows),
        "n_valid": len(valid_rows),
        "valid_ratio": float(len(valid_rows) / len(rows)) if rows else 0.0,
        "sa_pass_threshold": float(sa_pass_threshold),
    }
    for key in (
        "qed",
        "sa",
        "logp",
        "mw",
        "tpsa",
        "fraction_csp3",
        "structural_alert_count",
        "lipinski_pass",
        "veber_pass",
        "sa_pass",
    ):
        value = _numeric_mean(valid_rows, key)
        if value is not None:
            summary[f"{key}_mean" if not key.endswith("_pass") else f"{key}_ratio"] = value

    admet_keys = sorted({key for row in valid_rows for key in row if key.startswith("admet_")})
    summary["admet_means"] = {
        key: value
        for key in admet_keys
        if (value := _numeric_mean(valid_rows, key)) is not None
    }
    return summary


def add_admet_composite_objectives(row: Dict[str, object]) -> None:
    def _values(names: Sequence[str]) -> List[float]:
        return [float(row[f"admet_{name}"]) for name in names if f"admet_{name}" in row]

    safety = _values(ADMET_SAFETY_ENDPOINTS)
    cyp = _values(ADMET_CYP_INHIBITION_ENDPOINTS)
    absorption = _values(ADMET_ABSORPTION_ENDPOINTS)
    if safety:
        row["obj_admet_safety"] = max(0.0, min(1.0, 1.0 - mean(safety)))
    if cyp:
        row["obj_cyp_inhibition_safety"] = max(0.0, min(1.0, 1.0 - mean(cyp)))
    if absorption:
        row["obj_admet_absorption"] = max(0.0, min(1.0, mean(absorption)))


def evaluate_smiles(
    smiles_list: List[str],
    sa_pass_threshold: float = 6.0,
    admet_backend: str = "none",
    admet_endpoints: str | Sequence[str] | None = "recommended",
    admet_num_workers: int | None = None,
) -> Dict[str, object]:
    per_molecule = [_calc_rdkit_props(smiles) for smiles in smiles_list]
    valid_rows = [row for row in per_molecule if int(row.get("valid", 0)) == 1]
    for row in valid_rows:
        row["sa_pass"] = int(float(row["sa"]) <= sa_pass_threshold)

    if admet_backend == "admet_ai" and valid_rows:
        canonical_smiles = [str(row["canonical_smiles"]) for row in valid_rows]
        predictions = _predict_admet_ai(canonical_smiles, admet_endpoints, admet_num_workers)
        for row in valid_rows:
            row.update(predictions.get(str(row["canonical_smiles"]), {}))
            add_admet_composite_objectives(row)
    elif admet_backend != "none":
        raise ValueError(f"Unsupported ADMET backend: {admet_backend}")

    return {
        "backend": {"rdkit": True, "admet": admet_backend},
        "summary": summarize_property_rows(per_molecule, sa_pass_threshold),
        "per_molecule": per_molecule,
    }


def evaluate_smiles_file(
    smiles_json_path: str,
    output_json_path: str,
    sa_pass_threshold: float = 6.0,
    admet_backend: str = "none",
    admet_endpoints: str | Sequence[str] | None = "recommended",
    admet_num_workers: int | None = None,
) -> Dict[str, object]:
    with open(smiles_json_path, "r", encoding="utf-8") as handle:
        smiles = json.load(handle)
    if not isinstance(smiles, list):
        raise ValueError("Input JSON must be a list of SMILES strings.")
    result = evaluate_smiles(
        [str(item) for item in smiles],
        sa_pass_threshold=sa_pass_threshold,
        admet_backend=admet_backend,
        admet_endpoints=admet_endpoints,
        admet_num_workers=admet_num_workers,
    )
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
