import logging
import os
from time import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from frag_gt.src.io import load_smiles_from_file, valid_mols_from_smiles
from frag_gt.src.mapelites import map_elites_factory
from frag_gt.src.population import MolecularPopulationGenerator, Molecule
from frag_gt.src.scorers import SmilesScorer

logger = logging.getLogger(__name__)


_DEFAULT_FRAGSTORE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/fragment_libraries")
_DEFAULT_POPULATION_SMILES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/smiles_files")
DEFAULT_FRAGSTORE_PATH = os.path.join(_DEFAULT_FRAGSTORE_DIR, "chembl_33_chemreps_std_fragstore_brics_filter2.pkl")
DEFAULT_STARTING_POPULATION_SMILES_PATH = os.path.join(_DEFAULT_POPULATION_SMILES_DIR, "chembl_33_chemreps_std.smiles")


def _dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    ge_all = all(float(a.get(key, 0.0)) >= float(b.get(key, 0.0)) for key in keys)
    gt_any = any(float(a.get(key, 0.0)) > float(b.get(key, 0.0)) for key in keys)
    return ge_all and gt_any


def _pareto_fronts(items: List[Dict[str, float]], objective_keys: List[str]) -> List[List[int]]:
    if not items:
        return []

    dominates = [set() for _ in items]
    dominated_count = [0 for _ in items]
    fronts: List[List[int]] = [[]]

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
        next_front: List[int] = []
        for i in fronts[f]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)

        f += 1

    return fronts


def _crowding_distance(front: List[int], items: List[Dict[str, float]], objective_keys: List[str]) -> Dict[int, float]:
    if not front:
        return {}

    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}

    for key in objective_keys:
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


def _select_by_pareto(
    population: List[Molecule],
    objective_rows: List[Dict[str, float]],
    objective_keys: List[str],
    n_select: int,
) -> List[Molecule]:
    if not population:
        return []

    fronts = _pareto_fronts(objective_rows, objective_keys)
    selected_indices: List[int] = []

    for front in fronts:
        if len(selected_indices) + len(front) <= n_select:
            selected_indices.extend(front)
        else:
            distances = _crowding_distance(front, objective_rows, objective_keys)
            remaining = n_select - len(selected_indices)
            front_sorted = sorted(front, key=lambda idx: distances.get(idx, 0.0), reverse=True)
            selected_indices.extend(front_sorted[:remaining])
            break

    return [population[idx] for idx in selected_indices]


class FragGTGenerator:
    def __init__(
        self,
        smi_file: str = DEFAULT_STARTING_POPULATION_SMILES_PATH,
        fragmentation_scheme: str = "brics",
        fragstore_path: str = DEFAULT_FRAGSTORE_PATH,
        allow_unspecified_stereo: bool = False,
        scorer: str = "counts",
        operators: Optional[List[Tuple[str, float]]] = None,
        population_size: int = 100,
        n_mutations: int = 50,
        generations: int = 500,
        map_elites: Optional[str] = None,
        random_start: bool = False,
        patience: int = 5,
        n_jobs: int = -1,
        intermediate_results_dir: Optional[str] = None,
        selection_strategy: str = "score",
        pareto_objective_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            selection_strategy:
                "score"  -> legacy scalar score sorting.
                "pareto" -> generation-level Pareto survival selection.
            pareto_objective_keys:
                objective keys used by Pareto selection.
        """
        self.smi_file = smi_file
        self.fragmentation_scheme = fragmentation_scheme
        self.fragstore_path = fragstore_path
        self.allow_unspecified_stereo = allow_unspecified_stereo
        self.scorer = scorer
        self.operators = operators
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.generations = generations
        self.random_start = random_start
        self.patience = patience
        self.n_jobs = n_jobs
        self.intermediate_results_dir = intermediate_results_dir
        self.selection_strategy = selection_strategy
        self.pareto_objective_keys = pareto_objective_keys

        if self.selection_strategy not in {"score", "pareto"}:
            raise ValueError("selection_strategy must be 'score' or 'pareto'.")

        self.map_elites = None if map_elites is None else map_elites_factory(map_elites, fragmentation_scheme)

        logger.info(self.__dict__)

    def get_initial_population(self, size: int) -> List[Chem.rdchem.Mol]:
        raw_smiles = load_smiles_from_file(self.smi_file)
        if self.random_start:
            logger.info(f"taking a random subset of smiles as initial population (init_size: {size})")
            raw_smiles = np.random.choice(raw_smiles, size)
        initial_population = valid_mols_from_smiles(raw_smiles, self.n_jobs)
        return initial_population

    @staticmethod
    def deduplicate(population: List[Molecule]) -> List[Molecule]:
        unique_smiles = set()
        unique_population = []
        for molecule in population:
            smiles = Chem.MolToSmiles(molecule.mol)
            if smiles not in unique_smiles:
                unique_population.append(molecule)
                unique_smiles.add(smiles)
        return unique_population

    def write_generation_results(
        self,
        population: List[Molecule],
        generation: int,
        job_name: str,
        all_generations_results_dir: str,
    ):
        df = pd.DataFrame()
        df["SMILES"] = [Chem.MolToSmiles(p.mol) for p in population]
        df["scores"] = [p.score for p in population]
        df["gen"] = generation + 1
        df["fragmentation"] = self.fragmentation_scheme
        df["fragstore"] = self.fragstore_path

        safe_job_name = job_name.replace(" ", "_").lower()
        df.to_csv(f"{all_generations_results_dir}/{str(safe_job_name)}_{generation + 1}.csv")

    def _score_molecules(self, scoring_function: SmilesScorer, mols: List[Chem.rdchem.Mol]) -> List[Molecule]:
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        scores = scoring_function.score_list(smiles)
        return [Molecule(*item) for item in zip(scores, mols)]

    def _select_population(
        self,
        population: List[Molecule],
        scoring_function: SmilesScorer,
    ) -> List[Molecule]:
        population = self.deduplicate(population)

        if self.map_elites is not None:
            population, _ = self.map_elites.place_in_map(population)

        if self.selection_strategy != "pareto":
            return sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

        if not hasattr(scoring_function, "objectives"):
            logger.warning("selection_strategy='pareto' requested, but scoring_function has no objectives(); falling back to score sorting.")
            return sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

        objective_keys = self.pareto_objective_keys
        if objective_keys is None:
            objective_keys = getattr(scoring_function, "pareto_objective_keys", None)

        if not objective_keys:
            logger.warning("No pareto_objective_keys found; falling back to score sorting.")
            return sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

        valid_population: List[Molecule] = []
        objective_rows: List[Dict[str, float]] = []

        for molecule in population:
            smiles = Chem.MolToSmiles(molecule.mol)
            try:
                objectives = scoring_function.objectives(smiles)
            except Exception as exc:
                logger.debug("Failed to compute Pareto objectives for %s: %s", smiles, exc)
                objectives = None

            if objectives is None:
                continue

            valid_population.append(molecule)
            objective_rows.append(objectives)

        if not valid_population:
            logger.warning("No valid Pareto objective rows; falling back to score sorting.")
            return sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

        selected = _select_by_pareto(
            population=valid_population,
            objective_rows=objective_rows,
            objective_keys=objective_keys,
            n_select=self.population_size,
        )

        selected = sorted(selected, key=lambda x: x.score, reverse=True)

        if len(selected) < self.population_size:
            selected_smiles = {Chem.MolToSmiles(mol.mol) for mol in selected}
            fallback = [
                mol for mol in sorted(population, key=lambda x: x.score, reverse=True)
                if Chem.MolToSmiles(mol.mol) not in selected_smiles
            ]
            selected.extend(fallback[: self.population_size - len(selected)])

        return selected[:self.population_size]

    def optimize(
        self,
        scoring_function: SmilesScorer,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
        fixed_substructure_smarts: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> List[str]:
        """
        Generate optimal molecules.

        In selection_strategy='pareto' mode, every generation uses Pareto
        survival selection instead of simple scalar-score sorting.
        """
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            logger.info(f"Benchmark requested more molecules than expected: new population is {number_molecules}")

        logger.info("preparing initial population...")
        if starting_population is None:
            logger.info(f"loading initial population from smiles file: {self.smi_file}")
            initial_population_size = (self.population_size + self.n_mutations) * 4
            initial_population = self.get_initial_population(size=initial_population_size)
        else:
            logger.info(f"using user provided initial population for generation: {starting_population}")
            initial_population = valid_mols_from_smiles(starting_population, self.n_jobs)

        logger.info("scoring initial population...")
        population = self._score_molecules(scoring_function, initial_population)
        population = self._select_population(population, scoring_function)

        if (self.intermediate_results_dir is not None) and (job_name is not None):
            self.write_generation_results(population, -1, job_name, self.intermediate_results_dir)

        population_scores = [p.score for p in population]

        mol_generator = MolecularPopulationGenerator(
            fragstore_path=self.fragstore_path,
            fragmentation_scheme=self.fragmentation_scheme,
            n_molecules=self.n_mutations,
            operators=self.operators,
            allow_unspecified_stereo=self.allow_unspecified_stereo,
            selection_method="tournament-3",
            scorer=self.scorer,
            fixed_substructure_smarts=fixed_substructure_smarts,
        )

        logger.info("starting evolution...")
        logger.info(
            f"i | max: {np.max(population_scores):.3f} | "
            f"avg: {np.mean(population_scores):.3f} | "
            f"min: {np.min(population_scores):.3f} | "
            f"std: {np.std(population_scores):.3f} | "
            f"pop: {len(population_scores)} | "
            f"selection: {self.selection_strategy}"
        )

        patience = 0
        t0 = time()

        for generation in range(self.generations):
            old_scores = population_scores

            new_population_mols = mol_generator.generate(population)

            existing_population_smiles = {Chem.MolToSmiles(x.mol) for x in population}
            new_population_tuples = {(mol, Chem.MolToSmiles(mol)) for mol in new_population_mols}
            new_mol_tuples = [
                (mol, smiles) for mol, smiles in new_population_tuples
                if smiles not in existing_population_smiles
            ]

            if not len(new_mol_tuples):
                patience += 1
                logger.info(f"Failed to progress by generating new molecules: {patience}")
                if patience >= self.patience:
                    logger.info("No more patience, bailing...")
                    break
                continue

            new_molecules, new_smiles = zip(*new_mol_tuples)
            logger.debug(f"{len(new_population_mols) - len(new_molecules)} smiles already existed in the population")

            new_scores = scoring_function.score_list(new_smiles)
            assert len(new_scores) == len(new_molecules)

            new_population = [Molecule(*item) for item in zip(new_scores, new_molecules)]

            population += new_population
            population = self._select_population(population, scoring_function)

            population_scores = [p.score for p in population]

            if population_scores == old_scores:
                patience += 1
                logger.info(f"Failed to progress on fitness landscape: {patience}")
                if patience >= self.patience:
                    logger.info("No more patience, bailing...")
                    break
            else:
                patience = 0

            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / max(gen_time, 1e-12)
            t0 = time()

            logger.info(
                f"{generation} | "
                f"max: {np.max(population_scores):.3f} | "
                f"avg: {np.mean(population_scores):.3f} | "
                f"min: {np.min(population_scores):.3f} | "
                f"std: {np.std(population_scores):.3f} | "
                f"pop: {len(population_scores)} | "
                f"{gen_time:.2f} sec/gen | "
                f"{mol_sec:.2f} mol/sec | "
                f"selection: {self.selection_strategy}"
            )

            if (self.intermediate_results_dir is not None) and (job_name is not None):
                self.write_generation_results(population, generation, job_name, self.intermediate_results_dir)

        return [Chem.MolToSmiles(molecule.mol) for molecule in population[:number_molecules]]