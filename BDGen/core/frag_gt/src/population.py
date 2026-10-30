import logging
import json
import os
from pathlib import Path
from collections import Counter
from bde_frag_gt.regions import prepare_parent, prepare_donor
from bde_frag_gt.diagnostics import DiagnosticRecorder
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from frag_gt.src.fragmentors import fragmentor_factory
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.operators import mc_operator_factory
from frag_gt.src.query_builder import FragQueryBuilder
from frag_gt.src.stereo import enumerate_unspecified_stereocenters

logger = logging.getLogger(__name__)
Molecule = namedtuple("Molecule", ["score", "mol"])


class MolecularPopulationGenerator:
    """ handles generation of a population of molecules, knows nothing of molecular fitness """
    def __init__(self,
                 fragstore_path: str,
                 fragmentation_scheme: str,
                 n_molecules: int,
                 operators: Optional[List[Tuple[str, float]]] = None,
                 allow_unspecified_stereo: bool = False,
                 selection_method: str = "tournament",
                 scorer: str = "counts",
                 fixed_substructure_smarts: Optional[str] = None,
                 patience: int = 1000, region_policy=None, audit_path=None,
                 heavy_atom_constraint=None):

        self.n_molecules = n_molecules
        self.region_policy = region_policy
        self.audit_path = audit_path
        self.heavy_atom_constraint = heavy_atom_constraint
        self.diagnostics = None
        if os.environ.get('BDGEN_DIAGNOSTICS') == '1':
            if not audit_path:
                raise ValueError('Diagnostic recording requires an audit/output path')
            diagnostic_folder = Path(audit_path).parent
            self.diagnostics = DiagnosticRecorder(
                diagnostic_folder,
                sample_limit=int(os.environ.get('BDGEN_DIAG_SAMPLE_LIMIT', '30')),
                metadata={'case_id': diagnostic_folder.parent.parent.name,
                          'strategy': diagnostic_folder.parent.name,
                          'repeat': diagnostic_folder.parent.parent.parent.name})
        self.trace = []
        self.fragmentor = fragmentor_factory(fragmentation_scheme)
        self._fragstore_path = fragstore_path
        self._fragstore = fragstore_factory("in_memory", fragstore_path)

        # the query builder supports the mutation operators and controls how we sample the fragstore
        self.fragstore_qb = FragQueryBuilder(self._fragstore,
                                             scorer=scorer,
                                             sort_by_score=False,
                                             single_frag_prob=0.02,
                                             sample_with_replacement=True)
        assert self._fragstore.scheme == self.fragmentor.name

        # tuple of (mutation/crossover operator, probability of applying)
        if operators is None:
            logger.debug("using default mutation and crossover operators")
            operators = [
                ("substitute_node_mutation", 0.4),
                ("add_node_mutation", 0.05),
                ("delete_node_mutation", 0.05),
                ("single_point_crossover", 0.45),
                ("substitute_edge_mutation", 0.05),
            ]
        logger.info(f"operator probabilities: {operators}")
        assert sum([tup[1] for tup in operators]) == 1.0, "operator probabilities must sum to one"
        self.mol_operators, self.mol_operator_probs = zip(*operators)
        self.allow_unspecified_stereo = allow_unspecified_stereo
        self.selection_method = selection_method

        # keep substructures fixed in the population
        self._fixed_substructure_smarts = fixed_substructure_smarts
        self.fixed_substructure = None
        if fixed_substructure_smarts is not None:
            self.fixed_substructure = Chem.MolFromSmarts(fixed_substructure_smarts)
            if self.fixed_substructure is None:
                raise ValueError(f"invalid smarts pattern returned None: {fixed_substructure_smarts}")
        self.patience = patience
        self.max_molecular_weight = 1500

    @staticmethod
    def tournament_selection(population: List[Molecule], k: int = 5) -> Molecule:
        """
        tournament selection randomly chooses k individuals from
        the population and returns the fittest one
        """
        entrant_idxs = np.random.choice(len(population), size=k, replace=True)
        fittest = sorted([population[idx] for idx in entrant_idxs], key=lambda x: x.score, reverse=True)[0]
        return fittest

    def generate(self, current_pool: List[Molecule], generation=0) -> List[Chem.rdchem.Mol]:
        """ generate a new pool of molecules """

        new_pool: List[Chem.rdchem.Mol] = []
        self.current_stats = Counter()
        self.current_rejections = Counter()
        if self.diagnostics is not None:
            self.diagnostics.start_generation(generation)
        if self.region_policy is not None and self.region_policy.selection == 'static_weakest':
            current_pool = [p for p in current_pool if self.region_policy.locate_static(p.mol) is not None]
        if not current_pool:
            self.trace.append({'generation':generation,'operator_attempts':0,'children':0,'reason':'no_editable_parents'})
            if self.diagnostics is not None:
                self.diagnostics.finish_generation()
            return []
        patience = 0
        while len(new_pool) < self.n_molecules:
            if self.current_stats['operator_attempts'] >= max(100,self.n_molecules*20):
                break
            self.current_stats['operator_attempts'] += 1

            # select molecule(s) from the current pool
            if self.selection_method == "random":
                idxs = np.random.choice(len(current_pool), size=2)
                choices = [current_pool[i] for i in idxs]
            elif self.selection_method.startswith("tournament"):
                tournament_size = 3 if not "-" in self.selection_method else int(self.selection_method.split("-")[-1])
                choice_1 = self.tournament_selection(current_pool, k=tournament_size)
                choice_2 = self.tournament_selection(current_pool, k=tournament_size)
                choices = [choice_1, choice_2]
            else:
                raise ValueError(f"Unrecognised selection method: {self.selection_method}")

            # select a molecular transform operator (i.e mutation, crossover)
            j = np.random.choice(len(self.mol_operator_probs), 1, p=self.mol_operator_probs)[0]
            mc_op = self.mol_operators[j]

            # apply operators to generate children from parent compounds
            parent_mol = prepare_parent(choices[0].mol)
            region = None
            if self.region_policy is not None and self.region_policy.selection != 'none':
                region = self.region_policy.target(parent_mol)
                if region is None:
                    self.current_rejections['unresolved_target'] += 1
                    if self.diagnostics is not None:
                        self.diagnostics.start_attempt(self.current_stats['operator_attempts'], mc_op,
                                                       parent_mol, None,
                                                       choices[1].mol if mc_op.endswith('crossover') else None)
                        self.diagnostics.count('attempts')
                        self.diagnostics.count('unresolved_targets')
                        self.diagnostics.sample('unresolved_target')
                        self.diagnostics.end_attempt()
                    continue
            if self.diagnostics is not None:
                self.diagnostics.start_attempt(self.current_stats['operator_attempts'], mc_op,
                                               parent_mol, region,
                                               choices[1].mol if mc_op.endswith('crossover') else None)
                self.diagnostics.count('attempts')
            operator_raised = False
            operator_exception_type = None
            try:
                if not mc_op.endswith("crossover"):
                    children = mc_operator_factory(mc_op)(parent_mol, self.fragmentor, self.fragstore_qb, region=region)
                else:
                    second_parent_mol = prepare_donor(choices[1].mol)
                    children = mc_operator_factory(mc_op)(parent_mol, second_parent_mol, self.fragmentor, region=region)
            except (ValueError,RuntimeError,IndexError,KeyError) as exc:
                self.current_rejections['operator_exception_'+type(exc).__name__] += 1
                operator_raised = True
                operator_exception_type = type(exc).__name__
                if self.diagnostics is not None:
                    self.diagnostics.count('operator_exceptions', reason=type(exc).__name__)
                    self.diagnostics.sample('operator_exception:'+type(exc).__name__,
                                            details={'exception_type':type(exc).__name__,
                                                     'exception_message':str(exc)})
                children = []
            if self.diagnostics is not None:
                self.diagnostics.operator_outcome(len(children), exception_type=operator_exception_type)
                self.diagnostics.count('raw_children', len(children))
                if not operator_raised:
                    self.diagnostics.count('operator_completed_attempts')
                    if not children:
                        self.diagnostics.count('operator_empty_returns')
                        self.diagnostics.sample('operator_empty_return')
                    else:
                        self.diagnostics.count('operator_produced_children_attempts')

            # handle unspecified stereocenters if necessary
            if not self.allow_unspecified_stereo:
                # todo check what happens if fragment has a terminal chiral group?
                explicit_stereo_children = []
                for child in children:
                    explicit_stereo_children.extend(enumerate_unspecified_stereocenters(child))
                children = explicit_stereo_children
            if self.diagnostics is not None:
                self.diagnostics.count('children_after_stereo', len(children))

            # filter molecules that do not maintain the fixed SMARTS
            # the advantage of filtering here instead of including a SMARTS scoring function
            # is that we dont waste compute on scoring functions
            if self.fixed_substructure is not None:
                filtered_children = []
                for mol in children:
                    if mol.HasSubstructMatch(self.fixed_substructure, useChirality=True):
                        filtered_children.append(mol)
                    elif self.diagnostics is not None:
                        self.diagnostics.count('fixed_substructure_rejected_children')
                        self.diagnostics.sample('fixed_substructure_rejected', child=mol)
                children = filtered_children

            # filter by mol weight (very large number, stops expensive calls to scoring fns)
            if self.max_molecular_weight > 0:
                filtered_children = []
                for i in range(len(children)):
                    if MolWt(children[i]) <= self.max_molecular_weight:
                        filtered_children.append(children[i])
                    else:
                        logger.info(f"Dropping molecule with MW exceeding max size ({self.max_molecular_weight})...")
                        if self.diagnostics is not None:
                            self.diagnostics.count('mw_rejected_children')
                            self.diagnostics.sample('mw_ceiling_exceeded', child=children[i],
                                                    details={'maximum_molecular_weight':self.max_molecular_weight})
                children = filtered_children

            if getattr(self, 'heavy_atom_constraint', None) is not None:
                retained = []
                for child in children:
                    self.current_stats['heavy_atom_checked_children'] += 1
                    if self.diagnostics is not None:
                        self.diagnostics.count('heavy_atom_checked_children')
                    if self.heavy_atom_constraint.accepts(child):
                        retained.append(child)
                        self.current_stats['heavy_atom_passed_children'] += 1
                        if self.diagnostics is not None:
                            self.diagnostics.count('heavy_atom_passed_children')
                    else:
                        self.current_stats['heavy_atom_rejected_children'] += 1
                        self.current_rejections['heavy_atoms_below_seed_minimum'] += 1
                        if self.diagnostics is not None:
                            self.diagnostics.count('heavy_atom_rejected_children',
                                                   reason='below_fixed_original_seed_minimum')
                            self.diagnostics.sample('heavy_atoms_below_seed_minimum', child=child,
                                details=dict(child_heavy_atoms=child.GetNumHeavyAtoms(),
                                             **self.heavy_atom_constraint.metadata()))
                children = retained

            if region is not None:
                accepted = []
                for child in children:
                    report = self.region_policy.validate(parent_mol,child,region)
                    if self.diagnostics is not None:
                        self.diagnostics.region_result(child, report)
                    if report['accepted']:
                        self.current_stats['accepted_targeted_edits'] += 1
                        self.current_stats['terminal_children'] += bool(report.get('static_terminal'))
                        accepted.append(child)
                        if self.audit_path:
                            record = dict(report,generation=generation,operator=mc_op,region=region,
                                          parent_smiles=Chem.MolToSmiles(parent_mol),child_smiles=Chem.MolToSmiles(child))
                            record['parent_smiles_atom_order'] = json.loads(parent_mol.GetProp('_smilesAtomOutputOrder'))
                            with open(self.audit_path,'a') as handle:handle.write(json.dumps(record)+'\n')
                    else:self.current_rejections[report['reason']] += 1
                children = accepted
            if self.diagnostics is not None:
                self.diagnostics.count('generation_accepted_children', len(children))
                if not children:
                    self.diagnostics.count('attempts_without_accepted_children')
                else:
                    self.diagnostics.count('attempts_with_accepted_children')
                self.diagnostics.end_attempt()

            # breaks infinite loop if no molecules can be generated
            if not len(children):
                patience += 1
                if not patience % 100:
                    logger.info(f"Failed to generate molecule: {patience}")
                if patience >= self.patience:
                    logger.info("Could not generate new pool of molecules, bailing...")
                    break
            else:
                patience = 0

            # extend the pool with new molecules
            new_pool.extend(children)

        self.trace.append(dict(generation=generation,children=len(new_pool),rejections=dict(self.current_rejections),**self.current_stats))
        if self.diagnostics is not None:
            self.diagnostics.finish_generation()
        return new_pool
