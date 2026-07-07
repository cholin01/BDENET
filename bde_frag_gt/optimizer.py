from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors

from frag_gt.src.scorers import SmilesScorer


def _import_sascorer():
    """
    Import RDKit Contrib SA_Score/sascorer.py.

    Standard Ertl-Schuffenhauer synthetic accessibility score:
        1 = easy to synthesize
        10 = difficult to synthesize
    """
    rdkit_dir = os.path.dirname(rdkit.__file__)

    candidate_paths = [
        os.path.join(rdkit_dir, "Contrib", "SA_Score"),
        os.path.join(os.path.dirname(rdkit_dir), "rdkit", "Contrib", "SA_Score"),
        os.path.join(os.path.dirname(rdkit_dir), "share", "RDKit", "Contrib", "SA_Score"),
        os.path.join(sys.prefix, "share", "RDKit", "Contrib", "SA_Score"),
    ]

    for path in candidate_paths:
        sascorer_path = os.path.join(path, "sascorer.py")
        if os.path.exists(sascorer_path):
            if path not in sys.path:
                sys.path.append(path)
            import sascorer
            return sascorer

    raise ImportError(
        "Cannot find RDKit Contrib SA_Score/sascorer.py. "
        "Please check your RDKit installation. "
        "Common path: $CONDA_PREFIX/share/RDKit/Contrib/SA_Score/sascorer.py"
    )


sascorer = _import_sascorer()


class BDEPredictor(Protocol):
    """Interface for any BDE predictor model."""

    def predict_bond_bde(self, mol: Chem.Mol) -> List[float]:
        """Return one BDE value per bond in kcal/mol."""


class HeuristicBDEPredictor:
    """
    Fallback predictor with chemistry-informed priors.

    This is intentionally lightweight so the full optimization pipeline
    can run out-of-the-box. Replace with your trained predictor in production.
    """

    BOND_TYPE_BASE: Dict[Chem.BondType, float] = {
        Chem.BondType.SINGLE: 75.0,
        Chem.BondType.DOUBLE: 120.0,
        Chem.BondType.TRIPLE: 170.0,
        Chem.BondType.AROMATIC: 110.0,
    }

    def predict_bond_bde(self, mol: Chem.Mol) -> List[float]:
        values: List[float] = []

        for bond in mol.GetBonds():
            base = self.BOND_TYPE_BASE.get(bond.GetBondType(), 70.0)
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            hetero_bonus = 6.0 if (
                a1.GetAtomicNum() not in (1, 6) or a2.GetAtomicNum() not in (1, 6)
            ) else 0.0
            ring_bonus = 4.0 if bond.IsInRing() else 0.0
            conjugation_bonus = 3.0 if bond.GetIsConjugated() else 0.0

            values.append(float(base + hetero_bonus + ring_bonus + conjugation_bonus))

        return values


@dataclass
class BDEOptimizationConfig:
    mode: str = "global"

    target_min_bde: float = 90.0
    target_mean_bde: float = 105.0
    weakest_bde_range: Optional[Tuple[float, float]] = None
    maximize_bde: bool = False
    weakest_non_ring_only: bool = True

    target_bond_idx: Optional[int] = None
    target_bond_range: Optional[Tuple[float, float]] = None
    maximize_target_bond: bool = False

    seed_smiles: Optional[str] = None
    min_similarity: float = 0.55

    logp_range: Tuple[float, float] = (0.0, 4.0)
    mw_range: Tuple[float, float] = (150.0, 550.0)
    tpsa_range: Tuple[float, float] = (0.0, 140.0)
    sa_range: Tuple[float, float] = (1.0, 6.0)

    min_qed: Optional[float] = None
    min_logp: Optional[float] = None
    min_mw: Optional[float] = None
    max_tpsa: Optional[float] = None
    max_sa: Optional[float] = None

    weight_bde_min: float = 0.35
    weight_bde_mean: float = 0.20
    weight_qed: float = 0.15
    weight_logp: float = 0.08
    weight_mw: float = 0.05
    weight_tpsa: float = 0.07
    weight_sa: float = 0.10
    weight_similarity: float = 0.15
    weight_target_bond: float = 0.20

    structural_filter_profile: str = "none"
    local_hot_max_bde: Optional[float] = None
    weight_structural_filter: float = 0.20
    hot_top_k: int = 3

    # two-stage BDE
    use_two_stage_bde: bool = True
    bde_top_k_per_batch: int = 40
    bde_top_fraction_per_batch: float = 0.25
    bde_min_cheap_score: float = 0.25
    max_bde_mw: float = 650.0
    max_bde_heavy_atoms: int = 70
    max_bde_bonds: int = 75
    allow_objectives_without_bde: bool = True


def _bounded_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / max(1.0, abs(low)))
    return max(0.0, 1.0 - (value - high) / max(1.0, abs(high)))


def _target_score(value: float, target: float, scale: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / max(1e-6, scale))


def _sa_score_proxy(mol: Chem.Mol) -> float:
    """
    Standard Ertl-Schuffenhauer synthetic accessibility score.

    Lower is better:
        1 = easy to synthesize
        10 = difficult to synthesize

    The function name is kept as _sa_score_proxy so existing calls do not need
    to be changed, but the internal implementation is now the standard SA score.
    """
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 10.0


class BDEPropertyScorer(SmilesScorer):
    """
    Two-stage multi-objective scorer.

    Stage 1:
        Cheap RDKit properties only:
        QED / LogP / MW / TPSA / SA / similarity / structural filter.

    Stage 2:
        Only top cheap candidates and not-too-large molecules get BDENET BDE prediction.

    This avoids running expensive 3D conformer generation and neural BDE inference
    for every generated molecule.
    """

    pareto_objective_keys = [
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
        "obj_similarity",
    ]

    def __init__(
        self,
        predictor: Optional[BDEPredictor] = None,
        config: Optional[BDEOptimizationConfig] = None,
    ):
        self.predictor = predictor or HeuristicBDEPredictor()
        self.config = config or BDEOptimizationConfig()
        self._fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._seed_fp = None

        self._cheap_cache: Dict[str, Optional[Dict[str, float]]] = {}
        self._bde_cache: Dict[str, Optional[List[float]]] = {}
        self._full_cache: Dict[str, Optional[Dict[str, float]]] = {}

        self.n_cheap_eval = 0
        self.n_bde_eval = 0
        self.n_bde_skip_large = 0
        self.n_bde_skip_low_cheap = 0
        self.n_bde_failed = 0

        if self.config.seed_smiles:
            seed = Chem.MolFromSmiles(self.config.seed_smiles)
            if seed is not None:
                self._seed_fp = self._fingerprint_generator.GetFingerprint(seed)

    def _canon(self, mol_or_smiles) -> Optional[str]:
        try:
            if isinstance(mol_or_smiles, str):
                mol = Chem.MolFromSmiles(mol_or_smiles)
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            return Chem.MolToSmiles(mol_or_smiles, isomericSmiles=True)
        except Exception:
            return None

    def _cheap_values(self, mol: Chem.Mol) -> Optional[Dict[str, float]]:
        """
        Fast RDKit-only properties.
        No 3D generation.
        No BDENET.
        """
        key = self._canon(mol)
        if key is None:
            return None

        if key in self._cheap_cache:
            return self._cheap_cache[key]

        try:
            qed = QED.qed(mol)
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            sa = _sa_score_proxy(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            n_bonds = mol.GetNumBonds()

            similarity = 1.0
            if self._seed_fp is not None:
                cand_fp = self._fingerprint_generator.GetFingerprint(mol)
                similarity = float(DataStructs.TanimotoSimilarity(self._seed_fp, cand_fp))

            structural_filter_score = 1.0
            if self.config.mode == "local" and self.config.structural_filter_profile != "none":
                structural_filter_score = 1.0

            values = {
                "qed": float(qed),
                "logp": float(logp),
                "mw": float(mw),
                "tpsa": float(tpsa),
                "sa": float(sa),
                "similarity": float(similarity),
                "heavy_atoms": float(heavy_atoms),
                "n_bonds": float(n_bonds),
                "structural_filter_ok": float(structural_filter_score),
                "structural_alert_count": 0.0,
            }

            self._cheap_cache[key] = values
            self.n_cheap_eval += 1
            return values

        except Exception:
            self._cheap_cache[key] = None
            return None

    def _cheap_score_from_values(self, values: Dict[str, float]) -> float:
        """
        Cheap score used for pre-ranking.
        This score does not include BDE.
        """
        c = self.config

        qed = values["qed"]
        logp = values["logp"]
        mw = values["mw"]
        tpsa = values["tpsa"]
        sa = values["sa"]
        similarity = values["similarity"]

        logp_score = _bounded_score(logp, c.logp_range[0], c.logp_range[1])
        mw_score = _bounded_score(mw, c.mw_range[0], c.mw_range[1])
        tpsa_score = _bounded_score(tpsa, c.tpsa_range[0], c.tpsa_range[1])
        sa_score = _bounded_score(sa, c.sa_range[0], c.sa_range[1])

        components = [
            (c.weight_qed, qed),
            (c.weight_logp, logp_score),
            (c.weight_mw, mw_score),
            (c.weight_tpsa, tpsa_score),
            (c.weight_sa, sa_score),
        ]

        if self._seed_fp is not None:
            components.append((c.weight_similarity, similarity))

        active_weight = sum(w for w, _ in components)
        score = sum(w * s for w, s in components) / max(active_weight, 1e-8)

        if self._seed_fp is not None and similarity < c.min_similarity:
            score -= min(0.7, (c.min_similarity - similarity) * 2.0)

        if c.max_tpsa is not None and tpsa > c.max_tpsa:
            score -= min(0.5, (tpsa - c.max_tpsa) / max(1.0, abs(c.max_tpsa)))

        if c.max_sa is not None and sa > c.max_sa:
            score -= min(0.5, (sa - c.max_sa) / max(1.0, abs(c.max_sa)))

        return float(max(0.0, min(1.0, score)))

    def _should_predict_bde(
        self,
        mol: Chem.Mol,
        cheap_values: Dict[str, float],
        cheap_score: float,
    ) -> bool:
        """
        Decide whether this molecule deserves expensive BDENET prediction.
        """
        c = self.config

        if not c.use_two_stage_bde:
            return True

        if cheap_score < c.bde_min_cheap_score:
            self.n_bde_skip_low_cheap += 1
            return False

        if cheap_values["mw"] > c.max_bde_mw:
            self.n_bde_skip_large += 1
            return False

        if cheap_values["heavy_atoms"] > c.max_bde_heavy_atoms:
            self.n_bde_skip_large += 1
            return False

        if cheap_values["n_bonds"] > c.max_bde_bonds:
            self.n_bde_skip_large += 1
            return False

        return True

    def _get_bde_values(
        self,
        mol: Chem.Mol,
        force: bool = False,
        cheap_values: Optional[Dict[str, float]] = None,
        cheap_score: Optional[float] = None,
    ) -> Optional[List[float]]:
        """
        Cached BDENET inference.
        """
        key = self._canon(mol)
        if key is None:
            return None

        if key in self._bde_cache:
            return self._bde_cache[key]

        if cheap_values is None:
            cheap_values = self._cheap_values(mol)
            if cheap_values is None:
                self._bde_cache[key] = None
                return None

        if cheap_score is None:
            cheap_score = self._cheap_score_from_values(cheap_values)

        if not force and not self._should_predict_bde(mol, cheap_values, cheap_score):
            self._bde_cache[key] = None
            return None

        try:
            bdes = self.predictor.predict_bond_bde(mol)
            if not bdes:
                self.n_bde_failed += 1
                self._bde_cache[key] = None
                return None

            if len(bdes) != mol.GetNumBonds():
                self.n_bde_failed += 1
                self._bde_cache[key] = None
                return None

            bdes = [float(x) for x in bdes]
            self._bde_cache[key] = bdes
            self.n_bde_eval += 1
            return bdes

        except Exception:
            self.n_bde_failed += 1
            self._bde_cache[key] = None
            return None

    def _merge_values(
        self,
        mol: Chem.Mol,
        cheap_values: Dict[str, float],
        bdes: Optional[List[float]],
    ) -> Dict[str, float]:
        """
        Merge cheap properties and optional BDE values.
        """
        values = dict(cheap_values)

        if bdes:
            if self.config.weakest_non_ring_only:
                non_ring_bdes = [
                    bde for bond, bde in zip(mol.GetBonds(), bdes) if not bond.IsInRing()
                ]
                weakest_pool = non_ring_bdes if non_ring_bdes else bdes
            else:
                weakest_pool = bdes

            values["min_bde"] = float(min(weakest_pool))
            values["mean_bde"] = float(sum(bdes) / len(bdes))
            values["has_bde"] = 1.0
        else:
            values["min_bde"] = 0.0
            values["mean_bde"] = 0.0
            values["has_bde"] = 0.0

        return values

    def _score_from_values(self, values: Dict[str, float], mol: Chem.Mol) -> float:
        c = self.config

        cheap_score = self._cheap_score_from_values(values)

        if values.get("has_bde", 0.0) < 0.5:
            return cheap_score

        min_bde = values["min_bde"]
        mean_bde = values["mean_bde"]

        if c.weakest_bde_range is not None:
            low, high = c.weakest_bde_range
            bde_min_score = _bounded_score(min_bde, low, high)
        elif c.maximize_bde:
            bde_min_score = min(1.0, min_bde / max(1.0, c.target_min_bde))
        else:
            bde_min_score = _target_score(min_bde, c.target_min_bde, scale=50.0)

        if c.maximize_bde:
            bde_mean_score = min(1.0, mean_bde / max(1.0, c.target_mean_bde))
        else:
            bde_mean_score = _target_score(mean_bde, c.target_mean_bde, scale=60.0)

        bde_weight = c.weight_bde_min + c.weight_bde_mean
        cheap_weight = (
            c.weight_qed
            + c.weight_logp
            + c.weight_mw
            + c.weight_tpsa
            + c.weight_sa
            + (c.weight_similarity if self._seed_fp is not None else 0.0)
        )

        bde_score = (
            c.weight_bde_min * bde_min_score
            + c.weight_bde_mean * bde_mean_score
        ) / max(bde_weight, 1e-8)

        total = (
            cheap_weight * cheap_score
            + bde_weight * bde_score
        ) / max(cheap_weight + bde_weight, 1e-8)

        return float(max(0.0, min(1.0, total)))

    def _objectives_from_values(self, values: Dict[str, float]) -> Dict[str, float]:
        c = self.config

        logp = values["logp"]
        mw = values["mw"]
        tpsa = values["tpsa"]
        sa = values["sa"]

        return {
            "obj_min_bde": float(values.get("min_bde", 0.0)),
            "obj_mean_bde": float(values.get("mean_bde", 0.0)),
            "obj_qed": float(values["qed"]),
            "obj_logp_window": 1.0 if c.logp_range[0] <= logp <= c.logp_range[1] else 0.0,
            "obj_mw_window": 1.0 if c.mw_range[0] <= mw <= c.mw_range[1] else 0.0,
            "obj_tpsa_window": 1.0 if c.tpsa_range[0] <= tpsa <= c.tpsa_range[1] else 0.0,
            "obj_sa_window": 1.0 if c.sa_range[0] <= sa <= c.sa_range[1] else 0.0,
            "obj_sa": 1.0 / (1.0 + max(0.0, float(sa))),
            "obj_structural_filter": float(values.get("structural_filter_ok", 1.0)),
            "obj_structural_alerts": 1.0 / (1.0 + float(values.get("structural_alert_count", 0.0))),
            "obj_similarity": float(values.get("similarity", 1.0)),
            "obj_has_bde": float(values.get("has_bde", 0.0)),
        }

    def _basic_values(
        self,
        mol: Chem.Mol,
        force_bde: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Main value function.

        By default:
            cheap first, BDE optional.
        """
        key = self._canon(mol)
        if key is None:
            return None

        if not force_bde and key in self._full_cache:
            return self._full_cache[key]

        cheap_values = self._cheap_values(mol)
        if cheap_values is None:
            return None

        cheap_score = self._cheap_score_from_values(cheap_values)

        bdes = self._get_bde_values(
            mol,
            force=force_bde,
            cheap_values=cheap_values,
            cheap_score=cheap_score,
        )

        values = self._merge_values(mol, cheap_values, bdes)

        if not force_bde:
            self._full_cache[key] = values

        return values

    def cheap_score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        values = self._cheap_values(mol)
        if values is None:
            return 0.0

        return self._cheap_score_from_values(values)

    def score(self, smiles: str) -> float:
        """
        Single-molecule score.

        This no longer always triggers BDE.
        It uses cheap score plus cached BDE if already available.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        values = self._basic_values(mol, force_bde=False)
        if values is None:
            return 0.0

        return self._score_from_values(values, mol)

    def score_list(self, smiles_list: List[str]) -> List[float]:
        """
        Batch scoring with two-stage BDE.

        1. Compute cheap score for all molecules.
        2. Select top candidates.
        3. Run BDE only on top candidates that pass size and cheap filters.
        4. Return final scores for all molecules.
        """
        mols: List[Optional[Chem.Mol]] = []
        cheap_values_list: List[Optional[Dict[str, float]]] = []
        cheap_scores: List[float] = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)

            if mol is None:
                cheap_values_list.append(None)
                cheap_scores.append(0.0)
                continue

            values = self._cheap_values(mol)
            cheap_values_list.append(values)
            cheap_scores.append(self._cheap_score_from_values(values) if values is not None else 0.0)

        if self.config.use_two_stage_bde and smiles_list:
            valid_indices = [
                idx for idx, (mol, values, score) in enumerate(
                    zip(mols, cheap_values_list, cheap_scores)
                )
                if mol is not None
                and values is not None
                and score >= self.config.bde_min_cheap_score
                and values["mw"] <= self.config.max_bde_mw
                and values["heavy_atoms"] <= self.config.max_bde_heavy_atoms
                and values["n_bonds"] <= self.config.max_bde_bonds
            ]

            valid_indices = sorted(
                valid_indices,
                key=lambda idx: cheap_scores[idx],
                reverse=True,
            )

            k_by_fraction = int(max(1, round(len(smiles_list) * self.config.bde_top_fraction_per_batch)))
            k = min(self.config.bde_top_k_per_batch, k_by_fraction, len(valid_indices))

            selected_for_bde = valid_indices[:k]

            for idx in selected_for_bde:
                mol = mols[idx]
                values = cheap_values_list[idx]
                score = cheap_scores[idx]

                if mol is None or values is None:
                    continue

                self._get_bde_values(
                    mol,
                    force=True,
                    cheap_values=values,
                    cheap_score=score,
                )

        elif not self.config.use_two_stage_bde:
            for idx, mol in enumerate(mols):
                if mol is None or cheap_values_list[idx] is None:
                    continue

                self._get_bde_values(
                    mol,
                    force=True,
                    cheap_values=cheap_values_list[idx],
                    cheap_score=cheap_scores[idx],
                )

        final_scores: List[float] = []

        for mol in mols:
            if mol is None:
                final_scores.append(0.0)
                continue

            values = self._basic_values(mol, force_bde=False)
            if values is None:
                final_scores.append(0.0)
            else:
                final_scores.append(self._score_from_values(values, mol))

        return final_scores

    def objectives(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Return Pareto objectives.

        If BDE has not been calculated for this molecule, still return cheap objectives.
        This avoids "No valid Pareto objective rows".
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        values = self._basic_values(mol, force_bde=False)
        if values is None:
            return None

        if values.get("has_bde", 0.0) < 0.5 and not self.config.allow_objectives_without_bde:
            return None

        return self._objectives_from_values(values)

    def objectives_list(self, smiles_list: List[str]) -> List[Optional[Dict[str, float]]]:
        """
        Batch objectives.
        Make sure score_list is called first so that top cheap candidates get BDE cached.
        """
        _ = self.score_list(smiles_list)
        return [self.objectives(smiles) for smiles in smiles_list]

    def debug_summary(self) -> Dict[str, int]:
        return {
            "n_cheap_eval": int(self.n_cheap_eval),
            "n_bde_eval": int(self.n_bde_eval),
            "n_bde_skip_large": int(self.n_bde_skip_large),
            "n_bde_skip_low_cheap": int(self.n_bde_skip_low_cheap),
            "n_bde_failed": int(self.n_bde_failed),
            "cheap_cache_size": int(len(self._cheap_cache)),
            "bde_cache_size": int(len(self._bde_cache)),
            "full_cache_size": int(len(self._full_cache)),
        }


def molecule_properties(smiles: str, predictor: Optional[BDEPredictor] = None) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid smiles: {smiles}")

    p = predictor or HeuristicBDEPredictor()
    bdes = p.predict_bond_bde(mol)

    return {
        "min_bde": min(bdes) if bdes else 0.0,
        "mean_bde": (sum(bdes) / len(bdes)) if bdes else 0.0,
        "qed": float(QED.qed(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "sa": float(_sa_score_proxy(mol)),
    }


def weakest_bond_info(smiles: str, predictor: Optional[BDEPredictor] = None) -> Dict[str, float | int | str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid smiles: {smiles}")

    p = predictor or HeuristicBDEPredictor()
    bdes = p.predict_bond_bde(mol)

    if not bdes:
        return {
            "weakest_bond_idx": -1,
            "weakest_bond_type": "NA",
            "weakest_bde": 0.0,
        }

    idx = min(range(len(bdes)), key=lambda i: bdes[i])
    bond = mol.GetBondWithIdx(idx)

    return {
        "weakest_bond_idx": int(idx),
        "weakest_bond_type": str(bond.GetBondType()),
        "weakest_bde": float(bdes[idx]),
    }


def bde_hot_bonds(
    smiles: str,
    predictor: Optional[BDEPredictor] = None,
    top_k: int = 3,
    max_bde: Optional[float] = None,
) -> List[Dict[str, float | int | bool]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid smiles: {smiles}")

    p = predictor or HeuristicBDEPredictor()
    bdes = p.predict_bond_bde(mol)

    rows: List[Dict[str, float | int | bool]] = []

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        bde = float(bdes[idx]) if idx < len(bdes) else 0.0

        rows.append(
            {
                "bond_idx": int(idx),
                "begin_atom_idx": int(bond.GetBeginAtomIdx()),
                "end_atom_idx": int(bond.GetEndAtomIdx()),
                "is_in_ring": bool(bond.IsInRing()),
                "bde": float(bde),
            }
        )

    rows = [row for row in rows if not row["is_in_ring"]]
    rows.sort(key=lambda row: row["bde"])

    if max_bde is not None:
        rows = [row for row in rows if float(row["bde"]) <= max_bde]

    return rows[:top_k]