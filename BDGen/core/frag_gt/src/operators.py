import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BreakBRICSBonds

from bde_frag_gt.regions import (PARENT, SEED, atom_signature, bond_signature,
                                choose_fragment, restrict_cuts, prepare_donor)
from bde_frag_gt.diagnostics import note_operator

from frag_gt.src.afp import renumber_frag_attachment_idxs, match_fragment_attachment_points
from frag_gt.src.fragmentors import FragmentorBase
from frag_gt.src.gene_type_utils import get_gene_type, get_haplotype_from_gene_frag, get_attachment_type_idx_pairs
from frag_gt.src.query_builder import FragQueryBuilder

logger = logging.getLogger(__name__)

DUMMY_SMARTS = Chem.MolFromSmarts("[#0]")
DUMMY_PLUS_ANCHOR_SMARTS = Chem.MolFromSmarts("[#0]~*")


def _has_unspecified_stereo(mol):
    """Conservatively preserve any later stereoisomer enumeration path."""
    try:
        from rdkit.Chem.rdchem import StereoSpecified
        return any(item.specified == StereoSpecified.Unspecified
                   for item in Chem.FindPotentialStereo(Chem.Mol(mol)))
    except Exception:
        return True  # An uncertain prescreen must fall back to the original path.


def _identity_prescreen_children(parent, children, region):
    """Remove only certain regional no-ops after all original operator draws.

    This saves the subsequent region-footprint validation, not construction.
    Unrestricted operators did not reject identity children, so remain intact.
    An unspecified stereocentre can yield new children when enumeration is on;
    leave such raw children to the unchanged population-level stereo handling.
    """
    if region is None or not children:
        return children
    parent_smiles = Chem.MolToSmiles(Chem.Mol(parent), isomericSmiles=True)
    kept = []
    for child in children:
        child_smiles = Chem.MolToSmiles(Chem.Mol(child), isomericSmiles=True)
        if child_smiles == parent_smiles and not _has_unspecified_stereo(child):
            note_operator('identity_prescreen_rejected', prescreened_child_count=1,
                          prescreen_stage='after_reconnection', child_smiles=child_smiles)
        else:
            kept.append(child)
    return kept


def _single_port_identity_key(fragment):
    """Compare the attached chemical fragment, independent of BRICS port type.

    The actual new connection bond type is checked separately. One unique dummy
    atom fixes the attachment position; real atom isotopes/maps and all stereo
    remain part of the isomeric key.
    """
    copied = Chem.Mol(fragment)
    ports = [atom for atom in copied.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(ports) != 1 or len(ports[0].GetNeighbors()) != 1:
        return None
    ports[0].SetIsotope(0)
    ports[0].SetAtomMapNum(0)
    return Chem.MolToSmiles(copied, isomericSmiles=True)


def _selected_crossover_is_identity(parent, cut, removed, donor_fragment,
                                    retained_port, donor_port, fragmentor):
    """Prove a chosen single-port exchange is a no-op, without another draw.

    This intentionally leaves stereo at either cut endpoint and any unspecified
    stereo to full construction. The original post-construction validation then
    decides those cases. Failure to prove identity never rejects a proposal.
    """
    try:
        if donor_fragment.GetNumHeavyAtoms() != len(removed):
            return False
        u, v = cut[0]
        connection_key = tuple(str(value) for value in sorted((int(retained_port), int(donor_port))))
        if fragmentor.recombination_rules[connection_key] != parent.GetBondBetweenAtoms(u, v).GetBondType():
            return False
        endpoints = {u, v}
        if any(parent.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               for i in endpoints):
            return False
        if any(bond.GetStereo() != Chem.BondStereo.STEREONONE and
               endpoints.intersection((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), *bond.GetStereoAtoms()))
               for bond in parent.GetBonds()):
            return False
        if _has_unspecified_stereo(parent):
            return False
        removed_side = next(i for i, endpoint in enumerate(cut[0]) if endpoint in removed)
        original_fragment = _single_cut_fragment(parent, cut, cut[0][removed_side],
                                                 str(cut[1][removed_side]))
        original_key = _single_port_identity_key(original_fragment)
        return original_key is not None and original_key == _single_port_identity_key(donor_fragment)
    except Exception:
        return False


def _diagnostic_parent_ids(mol):
    """Read provenance only; diagnostic metadata must not mask an exception."""
    try:
        return [a.GetIntProp('_bdgen_parent_atom') for a in mol.GetAtoms()
                if a.GetAtomicNum() != 0 and a.HasProp('_bdgen_parent_atom')]
    except Exception:
        return []


def _diagnostic_fragments(frags):
    try:
        if not isinstance(frags, (list, tuple)):
            return {'input_type': type(frags).__name__}
        return [{'atoms': frag.GetNumAtoms(),
                 'parent_ids': _diagnostic_parent_ids(frag)} for frag in frags]
    except Exception:
        return {'metadata_unavailable': True}


def _diagnostic_get_frags(fragmentor, parent_mol):
    try:
        return fragmentor.get_frags(parent_mol)
    except Exception as exc:
        note_operator('fragmentation_failure', exception_type=type(exc).__name__,
                      parent_ids=_diagnostic_parent_ids(parent_mol))
        raise


def connect_mol_from_frags(frags: List[Chem.rdchem.Mol], fragmentor: FragmentorBase) -> Chem.rdchem.Mol:
    """
    Given a list of fragments (RDKit mol objects) with attachment points [*] marked by integer pairs (attachment_idx)
    Return a new mol object

    Atom properties are maintained from input frags

    Note: BRICSBuild function in rdkit enumerates results, is slow, and only considers cut_type (not attachment_idx)
    """
    _diagnostic_stage = 'combine_fragments'
    try:
        # combine mols to single object
        composite_frags = frags[0]
        for i in range(len(frags) - 1):
            composite_frags = Chem.CombineMols(composite_frags, frags[i + 1])

        # get pairs to be bonded together
        # dict keys are attachment_idxs. A pair of * atoms should both have this attachment_idx in the input frags
        # dict values are two-value tuple of lists. ([], [])
        # - the first list contains mol object atom idxs to be bonded
        # - the second list describes the type of re-connection that should be made (used to select bond type)
        # e.g. pairs = {3: [[0, 15], [3, 3]], 1: [[8, 17], [1, 1]], 2: [[21, 30], [7, 7]]}
        _diagnostic_stage = 'collect_attachment_pairs'
        pairs = {}  # type: Dict[int, Tuple[List[int], List[str]]]
        for match in composite_frags.GetSubstructMatches(DUMMY_PLUS_ANCHOR_SMARTS):
            dummy_atom = composite_frags.GetAtomWithIdx(match[0])
            anchor_atom = composite_frags.GetAtomWithIdx(match[1])

            attachment_idx = dummy_atom.GetProp("attachment_idx")
            new_bond_info = pairs.get(attachment_idx, ([], []))
            new_bond_info[0].append(anchor_atom.GetIdx())
            new_bond_info[1].append(dummy_atom.GetIsotope())
            new_bond_info[1].sort()  # sort isotopes ascending to match keys in fragmentor.recombination_rules
            pairs[attachment_idx] = new_bond_info

        # get dict describing {cut_type: bond type} relationships
        _diagnostic_stage = 'get_recombination_rules'
        recombination_dict = fragmentor.recombination_rules

        # make new bonds and delete bonds to [*]
        _diagnostic_stage = 'create_editable_composite'
        new_mol = Chem.RWMol(composite_frags)
        _diagnostic_stage = 'add_reconnection_bonds'
        for i in pairs:
            new_mol.AddBond(pairs[i][0][0], pairs[i][0][1], recombination_dict[(str(pairs[i][1][0]), str(pairs[i][1][1]))])
        _diagnostic_stage = 'extract_reconnected_molecule'
        new_mol = new_mol.GetMol()
        _diagnostic_stage = 'remove_dummy_atoms'
        new_mol = AllChem.DeleteSubstructs(new_mol, DUMMY_SMARTS)

        _diagnostic_stage = 'sanitize_reconnected_molecule'
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception as exc:
        note_operator('construction_failure', exception_type=type(exc).__name__,
                      construction_stage=_diagnostic_stage,
                      fragment_metadata=_diagnostic_fragments(frags),
                      fragments=frags, candidate=locals().get('new_mol'))
        raise


def _find_partner_frag_and_atom_idx(frag_list: List[Chem.rdchem.Mol], query_attachment_idx: int):
    """ Given a query_attachment_idx, find the corresponding fragment and atom that partners it in a list of frags """
    for n, frag in enumerate(frag_list):
        for m in frag.GetSubstructMatches(DUMMY_SMARTS):
            a = frag.GetAtomWithIdx(m[0])
            if a.GetProp("attachment_idx") == query_attachment_idx:
                partner_frag_idx = n
                partner_atom_idx = a.GetIdx()
                return partner_frag_idx, partner_atom_idx


def _find_pendant_frag_idxs(frags: List[Chem.rdchem.Mol]) -> List[int]:
    """ identify pendant frags (frags with only one attachment) """
    return [n for n, f in enumerate(frags) if len(get_gene_type(f).split("#")) == 1]


def _get_partner_cut(added_cut_type_new_frag: str, fragmentor: FragmentorBase) -> str:
    """
    given a cut type, find corresponding partner to join
    handles asymmetric cut types e.g. [*5]-[7*])
    returns a random join picked from available options
    """
    possible_joins = [rule for rule in fragmentor.recombination_rules if added_cut_type_new_frag in rule]
    random.shuffle(possible_joins)
    if not possible_joins:
        note_operator('no_partner_cut', cut_type=added_cut_type_new_frag)
    picked_join = possible_joins.pop(0)
    if picked_join[0] == added_cut_type_new_frag:
        return picked_join[1]
    else:
        return picked_join[0]


def _delete_boundary_ports(mol):
    """Return uniquely labelled, single-neighbor ports, or fail conservatively."""
    ports = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        if not atom.HasProp('attachment_idx') or atom.GetDegree() != 1:
            return None
        label = atom.GetProp('attachment_idx')
        if label in ports:
            return None
        neighbor = atom.GetNeighbors()[0]
        if neighbor.GetAtomicNum() == 0:
            return None
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        ports[label] = (atom.GetIdx(), neighbor.GetIdx(), atom.GetIsotope(), bond_signature(bond))
    return ports


def _delete_port_graph_key(mol, ports):
    """Exact chemical graph with each existing attachment label fixed in place."""
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
    for color, label in enumerate(sorted(ports), 1):
        copied.GetAtomWithIdx(ports[label][0]).SetAtomMapNum(color)
    return Chem.MolToSmiles(copied, canonical=True, isomericSmiles=True)


def _track_delete_boundary(query_frag, mutant_frag, region):
    """Recover only proven deletion-boundary identities, never a guessed MCS.

    A complete port-labelled chemical isomorphism can reuse the original fragment.
    Otherwise only allowed boundary endpoints with one-to-one existing-port
    correspondence inherit provenance. Protected and interior new atoms remain
    unmarked, so the unchanged region validator still rejects genuine outside edits.
    """
    mutant = Chem.Mol(mutant_frag)
    for atom in mutant.GetAtoms():
        for prop in (PARENT, SEED):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
    query_ports, mutant_ports = _delete_boundary_ports(query_frag), _delete_boundary_ports(mutant)
    if query_ports is None or mutant_ports is None or set(query_ports) != set(mutant_ports):
        note_operator('delete_boundary_unresolved', failure=False, cause='nonunique_or_mismatched_ports')
        return mutant
    if _delete_port_graph_key(query_frag, query_ports) == _delete_port_graph_key(mutant, mutant_ports):
        note_operator('delete_boundary_exact_fragment_reused', failure=False,
                      parent_ids=[a.GetIntProp(PARENT) for a in query_frag.GetAtoms() if a.HasProp(PARENT)])
        return Chem.Mol(query_frag)

    allowed = set(region['inside']) | set(region['anchors'])
    forward, reverse, consistent = defaultdict(set), defaultdict(set), {}
    for label, (_, old_idx, old_type, old_bond) in query_ports.items():
        _, new_idx, new_type, new_bond = mutant_ports[label]
        forward[old_idx].add(new_idx)
        reverse[new_idx].add(old_idx)
        old_atom, new_atom = query_frag.GetAtomWithIdx(old_idx), mutant.GetAtomWithIdx(new_idx)
        compatible = (old_type == new_type and old_bond == new_bond
                      and atom_signature(old_atom) == atom_signature(new_atom))
        pair = (old_idx, new_idx)
        consistent[pair] = consistent.get(pair, True) and compatible
    tracked = []
    for old_idx, new_indices in forward.items():
        if len(new_indices) != 1:
            continue
        new_idx = next(iter(new_indices))
        if len(reverse[new_idx]) != 1 or not consistent[(old_idx, new_idx)]:
            continue
        old_atom, new_atom = query_frag.GetAtomWithIdx(old_idx), mutant.GetAtomWithIdx(new_idx)
        if not old_atom.HasProp(PARENT) or old_atom.GetIntProp(PARENT) not in allowed:
            continue
        for prop in (PARENT, SEED):
            if old_atom.HasProp(prop):
                new_atom.SetIntProp(prop, old_atom.GetIntProp(prop))
        tracked.append(old_atom.GetIntProp(PARENT))
    note_operator('delete_boundary_ports_tracked' if tracked else 'delete_boundary_unresolved',
                  failure=False, tracked_parent_ids=sorted(tracked),
                  cause='existing_unique_compatible_ports' if tracked else 'no_proven_allowed_endpoint')
    return mutant


def delete_node_mutation(parent_mol: Chem.rdchem.Mol,
                         fragmentor: FragmentorBase,
                         frag_db: FragQueryBuilder,
                         strict: bool = True, region=None) -> List[Chem.rdchem.Mol]:
    """
    Delete a random pendant node
    (if strict=True) Check that severed "hanging" node exists in fragstore, else mutate to one that does
    (if strict=False) patch the hanging edge to hydrogen
    """
    # fragment parent
    frags = _diagnostic_get_frags(fragmentor, parent_mol)

    # if mol could not be fragmented, return empty list
    if len(frags) <= 1:
        note_operator('delete_unfragmentable', fragment_count=len(frags))
        logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} could not be fragmented")
        return []

    # identify pendant frags
    pendant_frag_idxs = _find_pendant_frag_idxs(frags)
    if not len(pendant_frag_idxs):
        note_operator('delete_no_pendant_fragment', fragment_count=len(frags))
        logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} has no pendant frags ")
        return []

    # choose a pendant fragment to remove
    chosen = choose_fragment(frags, region, pendant_frag_idxs)
    if chosen is None:
        note_operator('delete_region_filtered', pendant_count=len(pendant_frag_idxs))
        return []
    deleted_frag = frags.pop(chosen)
    note_operator('delete_selected_fragment', failure=False,
                  parent_ids=_diagnostic_parent_ids(deleted_frag))

    # get attachment_idx of pendant frag and use to locate remaining hanging edge
    a = deleted_frag.GetAtomWithIdx(deleted_frag.GetSubstructMatch(DUMMY_SMARTS)[0])
    hanging_idx = a.GetProp("attachment_idx")
    remaining_frag_idx, hanging_edge_idx = _find_partner_frag_and_atom_idx(frags, hanging_idx)

    # modify remaining frag to remove hanging attachment point
    remaining_frag = frags.pop(remaining_frag_idx)
    remaining_frag = Chem.RWMol(remaining_frag)
    remaining_frag.ReplaceAtom(hanging_edge_idx, Chem.Atom(1))
    remaining_frag = remaining_frag.GetMol()
    remaining_frag = Chem.RemoveHs(remaining_frag)
    Chem.SanitizeMol(remaining_frag)

    if not strict:
        # todo neaten
        accept_mutant = True
        mutant_frag = remaining_frag
    else:
        # check if frag exists in db
        # todo change (1) and (2) into functions
        # (1) get gene_type results from DB
        gene_type = get_gene_type(remaining_frag)
        gene_type_results = list(frag_db.db.get_records("gene_types", {"gene_type": gene_type}))

        # (2) check for specific gene among results
        if len(gene_type_results) > 1:
            note_operator('delete_multiple_library_gene_types', gene_type=gene_type,
                          result_count=len(gene_type_results))
            raise RuntimeError(f"MutationDelNode: DB is corrupted, multiple results for gene_type {gene_type}")
        elif gene_type == "":
            note_operator('delete_strict_empty_gene_type')
            # if there are only two frags in original mol and we are deleting one of them gene type is ""
            # hard to check so false here (would be okay in strict=False)
            logger.debug(f"MutationDelNode: deleting frag from {Chem.MolToSmiles(parent_mol)} "
                         f"only leaves one (strict behaviour)")
            frag_exists = False
        elif len(gene_type_results) == 0:
            note_operator('delete_strict_gene_type_missing', gene_type=gene_type)
            # this gene_type is not in the database
            logger.debug(f"MutationDelNode: hanging frag gene type ({gene_type}) not in DB (strict behaviour)")
            frag_exists = False
        else:
            # check if SMILES exist in frag store
            rf_haplotype_smi = Chem.MolToSmiles(get_haplotype_from_gene_frag(remaining_frag))
            rf_gene_smi = Chem.MolToSmiles(remaining_frag)

            frag_exists = False
            if rf_haplotype_smi in gene_type_results[0]["haplotypes"]:
                if rf_gene_smi in gene_type_results[0]["haplotypes"][rf_haplotype_smi]["gene_frags"]:
                    frag_exists = True
            if not frag_exists:
                note_operator('delete_strict_fragment_missing', gene_type=gene_type)
                logger.debug(f"MutationDelNode: mutated frag {rf_gene_smi} NOT found in DB (strict behaviour)")

        # (3) if mutated_frag does not exist, mutate to something that does
        if frag_exists:
            mutant_frag = remaining_frag
            accept_mutant = True
            note_operator('delete_strict_existing_fragment', failure=False,
                          parent_ids=_diagnostic_parent_ids(mutant_frag))
        else:
            query_frag = remaining_frag

            gene_type = get_gene_type(query_frag)  # todo we already have this above...
            mutant_smiles, mutant_scores = frag_db.query_frags(gene_type, query_frag, x_choices=0.3)
            mutant_smiles = [x for _, x in sorted(zip(mutant_scores, mutant_smiles), key=lambda pair: pair[0])]
            if not len(mutant_smiles):
                note_operator('delete_fallback_no_library', gene_type=gene_type)
                logger.debug(f"MutationDelNode: no frags with {gene_type} to mutate hanging frag to")
                return []

            accept_mutant = False
            while (accept_mutant is False) and (len(mutant_smiles) > 0):
                mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

                # align attachment_idxs of mutant frag to original reference
                mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

                # check reaction types agree
                query_idx_types = get_attachment_type_idx_pairs(query_frag)
                mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
                if query_idx_types != mutant_idx_types:
                    note_operator('delete_interface_mismatch')
                    accept_mutant = False
                    logger.debug(f"MutationDelNode: Not matching types: {query_idx_types} {mutant_idx_types}")
                    logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                                 f"{Chem.MolToSmiles(mutant_frag)}")
                else:
                    accept_mutant = True
                    if region is not None:
                        mutant_frag = _track_delete_boundary(query_frag, mutant_frag, region)
                    note_operator('delete_strict_fallback_selected', failure=False,
                                  replaced_parent_ids=_diagnostic_parent_ids(query_frag),
                                  replacement_parent_ids=_diagnostic_parent_ids(mutant_frag))

    if accept_mutant:
        # reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        note_operator('delete_exhausted')
        logger.debug(f"MutationDelNode: could not delete node from {Chem.MolToSmiles(parent_mol)}")
        return []


def substitute_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                             frag_db: FragQueryBuilder, strict: bool = True, region=None) -> List[Chem.rdchem.Mol]:
    """
    Choose a fragment from the parent molecule (fragmented using fragmentor)
    Retrieve fragments from the fragment store that have the same connectivity (gene_type)
    Align the new mutant fragment to the original fragment using the alignment fingerprint (afp)
    Reconnect the molecule incorporating the mutant fragment in place of the query fragment
    """

    # fragment parent molecule
    frags = _diagnostic_get_frags(fragmentor, parent_mol)

    # choose a fragment to mutate
    chosen = choose_fragment(frags, region)
    if chosen is None:
        note_operator('substitute_no_fragments' if not frags else 'substitute_region_filtered',
                      fragment_count=len(frags))
        return []
    query_frag = frags.pop(chosen)
    gene_type = get_gene_type(query_frag)

    attempt = -1
    accept_mutant = False
    while (attempt < 5) and (accept_mutant is False):
        attempt += 1

        # retrieve new fragment from fragment store via tournament selection
        mutant_smiles, mutant_scores = frag_db.query_frags(gene_type, query_frag, x_choices=0.3)
        if not len(mutant_scores):
            note_operator('substitute_no_library', gene_type=gene_type, attempt=attempt)
            logger.debug(f'MutationSubNode: nothing retrieved from db for {gene_type}')
            continue
        winning_mutant_smiles = mutant_smiles[np.argmax(mutant_scores)]
        mutant_frag = Chem.MolFromSmiles(winning_mutant_smiles)

        # mutant frag must be different from query
        if Chem.MolToSmiles(mutant_frag) == Chem.MolToSmiles(query_frag):
            note_operator('substitute_identity_candidate', attempt=attempt)
            logger.debug(f'MutationSubNode: skipping frag as its same as query')
            continue

        # align attachment_idxs of mutant frag to original reference
        mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

        # check reaction types agree
        query_idx_types = get_attachment_type_idx_pairs(query_frag)
        mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
        if query_idx_types != mutant_idx_types:
            note_operator('substitute_interface_mismatch', attempt=attempt)
            accept_mutant = False
            logger.debug(f"MutationSubNode: Not matching types: {query_idx_types} {mutant_idx_types}")
            logger.debug(f"MutationSubNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                         f"{Chem.MolToSmiles(mutant_frag)}")
        else:
            accept_mutant = True

    if accept_mutant:
        # reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return _identity_prescreen_children(parent_mol, [new_mol], region)
    else:
        note_operator('substitute_exhausted')
        logger.debug(f"MutationSubNode: no substitutions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def add_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                      frag_db: FragQueryBuilder, strict: bool = True, region=None) -> List[Chem.rdchem.Mol]:
    """
    Choose a random node to mutate
    Mutate to node with one too many attachment points
    Add a small attachment to the extra limb
    """

    # fragment parent
    frags = _diagnostic_get_frags(fragmentor, parent_mol)

    # choose a random fragment to mutate
    chosen = choose_fragment(frags, region)
    if chosen is None:
        note_operator('add_no_fragments' if not frags else 'add_region_filtered',
                      fragment_count=len(frags))
        return []
    query_frag = frags.pop(chosen)

    # Get query fragment gene_type
    gene_type = get_gene_type(query_frag)

    # create list of all possible cut types to add (in a random order)
    cut_types = fragmentor.get_cut_list(randomize_order=True)
    if not cut_types:
        note_operator('add_no_cut_types')

    # try to mutate gene_type until a suitable mutation is found
    gene_type_list = gene_type.split("#")
    accept_mutant = False
    while (accept_mutant is False) and (len(cut_types) > 0):
        # try to add new node with cut_type: added_cut_type_new_frag
        added_cut_type_new_frag = str(cut_types.pop(0))

        # retrieve small node for new pendant attachment
        # if no suitable nodes exist, continue to next cut type
        pendant_smiles_list, _ = frag_db.query_frags(added_cut_type_new_frag,
                                                     Chem.MolFromSmiles(f"[{added_cut_type_new_frag}*]C"),
                                                     x_choices=0.3)
        if len(pendant_smiles_list):
            # tournament selection for shortest smiles seq
            new_pfrag_smiles = min(pendant_smiles_list, key=len)
        else:
            note_operator('add_no_library_pendant', cut_type=added_cut_type_new_frag)
            logger.debug(f"MutationAddNode: No small fragments in DB with cut_type: {added_cut_type_new_frag}")
            continue

        # mutate existing fragment to contain an extra attachment point
        added_cut_type_existing_frag = _get_partner_cut(added_cut_type_new_frag, fragmentor)

        # given new cut type, try to mutate existing fragment
        mutant_gene_type_list = sorted([int(x) for x in gene_type_list if x is not ''] +
                                       [int(added_cut_type_existing_frag)])
        mutant_gene_type = "#".join([str(x) for x in mutant_gene_type_list])
        # logger.debug(f"MutationAddNode: Attempting to mutate existing gene {gene_type} -> {mutant_gene_type}")

        # query database to retrieve frags belonging to mutant_gene_type
        mutant_smiles, mutant_scores = frag_db.query_frags(mutant_gene_type, query_frag, x_choices=0.3)
        mutant_smiles = [x for _, x in sorted(zip(mutant_scores, mutant_smiles), key=lambda pair: pair[0])]
        if not mutant_smiles:
            note_operator('add_no_library_existing_fragment', gene_type=mutant_gene_type)

        while (accept_mutant is False) and (len(mutant_smiles) > 0):
            # try best first
            mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

            # align attachment_idxs of mutant frag to original reference
            mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

            # check aligned query and mutant cut-types agree, else continue to next candidate gene
            query_idx_types = get_attachment_type_idx_pairs(query_frag)
            mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
            difference = mutant_idx_types - query_idx_types
            if len(difference) == 1 and int(difference.pop()[1]) == -1:
                # ignore attachment id -1 since this is the newly added edge
                accept_mutant = True
            else:
                accept_mutant = False
                note_operator('add_interface_mismatch')
                logger.debug(f"MutationAddNode: Not matching types: {query_idx_types} {mutant_idx_types}")
                logger.debug(f"MutationAddNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                             f"{Chem.MolToSmiles(mutant_frag)}")

    if accept_mutant:
        # format added pendant frag
        new_pfrag = Chem.MolFromSmiles(new_pfrag_smiles)
        a = new_pfrag.GetAtomWithIdx(new_pfrag.GetSubstructMatch(DUMMY_SMARTS)[0])
        a.SetProp("attachment_idx", str(-1))

        # reconstruct list of frags for new molecule
        frags.extend([new_pfrag, mutant_frag])

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        note_operator('add_exhausted')
        logger.debug(f"MutationAddNode: no suitable additions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def substitute_edge_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                             frag_db: FragQueryBuilder, strict: bool = True, region=None) -> List[Chem.rdchem.Mol]:
    """
    Select pendant fragment and mutate to a different gene type
    Also mutate the gene type of the existing fragment to which the above was attached (the hanging frag)
    """

    # fragment parent molecule
    frags = _diagnostic_get_frags(fragmentor, parent_mol)
    if len(frags) == 1:
        note_operator('edge_unfragmentable', fragment_count=len(frags))
        return []

    # identify pendant frags
    pendant_frag_idxs = _find_pendant_frag_idxs(frags)
    if not len(pendant_frag_idxs):
        note_operator('edge_no_fragments' if not frags else 'edge_no_pendant_fragment',
                      fragment_count=len(frags))
        return []

    # choose a pendant frag to mutate gene type
    chosen = choose_fragment(frags, region, pendant_frag_idxs)
    if chosen is None:
        note_operator('edge_region_filtered', pendant_count=len(pendant_frag_idxs))
        return []
    pfrag = frags.pop(chosen)
    pfrag_gene_type = get_gene_type(pfrag)

    # get attachment_idx of pendant frag and use to locate remaining hanging edge
    a = pfrag.GetAtomWithIdx(pfrag.GetSubstructMatch(DUMMY_SMARTS)[0])
    hanging_idx = a.GetProp("attachment_idx")
    remaining_frag_idx, hanging_edge_idx = _find_partner_frag_and_atom_idx(frags, hanging_idx)

    # get hanging frag
    hfrag = frags.pop(remaining_frag_idx)
    # hfrag_gene_type = get_gene_type(hfrag)
    # todo used in strict setup to check and if not exist, mutate hfrag to real frag, same as delete

    # mutate hanging and pendant
    # could try multiple mutations here in a WHILE loop, but instead we simplify by randomizing and try first
    # todo should restrict to gene types with the same bond connection (possible joins), instead try except for now
    cut_types = fragmentor.get_cut_list(randomize_order=True)
    if not cut_types:
        note_operator('edge_no_cut_types')
    elif pfrag_gene_type not in cut_types:
        note_operator('edge_current_cut_type_missing', gene_type=pfrag_gene_type)
    cut_types.remove(pfrag_gene_type)
    if not cut_types:
        note_operator('edge_no_alternative_cut_type', gene_type=pfrag_gene_type)
    new_pcut_type = cut_types.pop(0)

    # mutate pfrag to a new gene type
    pmutant_smiles, pmutant_scores = frag_db.query_frags(new_pcut_type, pfrag, x_choices=0.3)
    if not len(pmutant_smiles):
        note_operator('edge_no_library_pendant', cut_type=new_pcut_type)
        logger.debug(f"MutationSubEdge: no pmutants found for {new_pcut_type}")
        # continue to next cut_type?
        return []
    winning_pmutant_smiles = pmutant_smiles[np.argmax(pmutant_scores)]

    # find corresponding pair types associated with `new_pcut_type`
    # mutate hanging fragment to substitute corresponding pair type in
    added_hcut_type = _get_partner_cut(new_pcut_type, fragmentor)

    # mutate old to new on both hfrag and pfrag
    a = hfrag.GetAtomWithIdx(hanging_edge_idx)
    att_idx = a.GetProp('attachment_idx')
    # logger.debug(f"MutationSubEdge: mutating hanging frag {a.GetIsotope()}->{added_hcut_type}")
    a.SetIsotope(int(added_hcut_type))
    # could mutate to an actually existing frag here? Strict flag? todo

    # annotate new pfrag with attachment_idx
    pmutant_frag = Chem.MolFromSmiles(winning_pmutant_smiles)
    a = pmutant_frag.GetAtomWithIdx(pmutant_frag.GetSubstructMatch(DUMMY_SMARTS)[0])
    a.SetProp('attachment_idx', att_idx)
    # logger.debug(f"MutationSubEdge: mutating pendant frag {pfrag_gene_type}->{new_pcut_type}")

    # reconstruct list of frags for new molecule
    frags.extend([hfrag, pmutant_frag])

    try:  # restrict gene types to same bond to remove this
        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        # does it need the accept malarky
        return _identity_prescreen_children(parent_mol, [new_mol], region)
    except Exception as e:
        note_operator('edge_construction_rejected', exception_type=type(e).__name__)
        logger.debug(f"MutationSubEdge: error {e}")
        logger.debug(f"MutationSubEdge: {Chem.MolToSmiles(parent_mol)} {winning_pmutant_smiles} {new_pcut_type}")
        return []


def _single_cut_sides(mol, cut):
    """Return endpoint-oriented atom sets without constructing candidate molecules."""
    (u, v), types = cut
    if mol.GetBondBetweenAtoms(u, v) is None:
        return None
    sides = []
    for start in (u, v):
        visited, pending = set(), [start]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                other = neighbor.GetIdx()
                if (current == u and other == v) or (current == v and other == u):
                    continue
                if other not in visited:
                    pending.append(other)
        sides.append(visited)
    if sides[0] & sides[1] or sides[0] | sides[1] != set(range(mol.GetNumAtoms())):
        return None  # A ring cut or disconnected input is not a two-piece exchange.
    return [(sides[0], u, str(types[0])), (sides[1], v, str(types[1]))]


def _single_cut_fragment(mol, cut, endpoint, expected_type):
    """Construct only the already selected side and verify its actual BRICS port."""
    broken = BreakBRICSBonds(mol, [cut])
    mappings = []
    fragments = list(Chem.GetMolFrags(broken, asMols=True, fragsMolAtomMapping=mappings))
    selected = [fragment for fragment, indices in zip(fragments, mappings) if endpoint in indices]
    if len(selected) != 1:
        raise ValueError('Selected crossover endpoint did not identify one fragment')
    fragment = renumber_frag_attachment_idxs(selected[0])
    dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1 or str(dummies[0].GetIsotope()) != expected_type:
        raise ValueError('Selected crossover fragment has an unexpected BRICS interface')
    return fragment


def _region_preserving_crossover(m1, m2, fragmentor, region, primary_cuts, donor_cuts):
    """Choose one feasible single-cut exchange; never retry candidate construction."""
    allowed = set(region['inside']) | set(region['anchors'])
    protected = set(range(m1.GetNumAtoms())) - allowed
    rules = fragmentor.recombination_rules
    donor_options = []
    for cut in donor_cuts:
        sides = _single_cut_sides(m2, cut)
        if sides is None:
            note_operator('region_crossover_donor_cut_filtered', failure=False,
                          cause='not_two_components', cut_atoms=list(cut[0]))
        else:
            donor_options.append((cut, sides))

    candidates = []
    for cut in primary_cuts:
        sides = _single_cut_sides(m1, cut)
        cause = None
        if sides is None:
            cause = 'not_two_components'
        else:
            directions = [i for i, (atoms, _, _) in enumerate(sides) if protected.issubset(atoms)]
            if not directions:
                cause = 'both_sides_protected'
            else:
                retained, endpoint, port_type = sides[directions[0]]
                removed = sides[1 - directions[0]][0]
                if not removed.issubset(allowed):
                    cause = 'removed_outside'
                elif endpoint not in allowed:
                    cause = 'outside_connection'
                else:
                    compatible = []
                    for donor_cut, donor_sides in donor_options:
                        # Retain the released operator's paired cut-type restriction.
                        if donor_cut[1] != cut[1]:
                            continue
                        for _, donor_endpoint, donor_port in donor_sides:
                            key = tuple(str(value) for value in sorted((int(port_type), int(donor_port))))
                            if key in rules:
                                compatible.append((donor_cut, donor_endpoint, donor_port))
                    if not compatible:
                        cause = 'no_compatible_donor_interface'
                    else:
                        candidates.append((cut, retained, removed, endpoint, port_type, compatible))
                        note_operator('region_crossover_feasible_primary_cut', failure=False,
                                      cut_atoms=list(cut[0]), retained_endpoint=endpoint,
                                      compatible_donor_option_count=len(compatible))
        if cause is not None:
            note_operator('region_crossover_cut_filtered_' + cause, failure=False,
                          cut_atoms=list(cut[0]), cut_types=list(cut[1]))

    note_operator('region_crossover_cut_inventory', failure=False,
                  considered_primary_cut_count=len(primary_cuts),
                  feasible_primary_cut_count=len(candidates),
                  filtered_primary_cut_count=len(primary_cuts) - len(candidates),
                  considered_donor_cut_count=len(donor_cuts))
    if not candidates:
        note_operator('region_crossover_no_feasible_cut',
                      primary_cut_count=len(primary_cuts), donor_cut_count=len(donor_cuts),
                      protected_parent_atoms=sorted(protected))
        return []

    # Exactly one draw for the primary cut and one for its compatible donor
    # direction. Failed construction propagates to the existing population
    # handler; no additional donor, cut, or region is tried in this attempt.
    cut, retained, removed, endpoint, port_type, compatible = random.choice(candidates)
    donor_cut, donor_endpoint, donor_port = random.choice(compatible)
    note_operator('region_crossover_selected_direction', failure=False,
                  primary_atoms=list(cut[0]), primary_types=list(cut[1]),
                  donor_atoms=list(donor_cut[0]), donor_types=list(donor_cut[1]),
                  retained_endpoint=endpoint, donor_endpoint=donor_endpoint,
                  retained_primary_atoms=sorted(retained), removed_primary_atoms=sorted(removed),
                  protected_parent_atoms=sorted(protected),
                  prospective_affected_parent_atoms=sorted(removed | {endpoint}),
                  feasible_primary_cut_count=len(candidates), compatible_donor_option_count=len(compatible))
    primary_fragment = _single_cut_fragment(m1, cut, endpoint, port_type)
    donor_fragment = prepare_donor(_single_cut_fragment(m2, donor_cut, donor_endpoint, donor_port))
    if _selected_crossover_is_identity(m1, cut, removed, donor_fragment,
                                       port_type, donor_port, fragmentor):
        note_operator('identity_prescreen_rejected', prescreened_child_count=1,
                      prescreen_stage='selected_single_port_before_reconnection',
                      child_smiles=Chem.MolToSmiles(Chem.Mol(m1), isomericSmiles=True))
        return []
    return _identity_prescreen_children(m1, [connect_mol_from_frags(
        [primary_fragment, donor_fragment], fragmentor=fragmentor)], region)


def single_point_crossover(m1: Chem.rdchem.Mol, m2: Chem.rdchem.Mol,
                           fragmentor: FragmentorBase, region=None) -> List[Chem.rdchem.Mol]:

    # get list of possible cuts for each molecules
    # tuple([pair of atom idxs], [pair of cut type str])
    # e.g. [([17, 16], ('5', '5')), ([5, 6], ('9', '9')), ([8, 7], ('9', '9'))]
    m1_possible_cuts = restrict_cuts(fragmentor.find_bonds(m1), region)
    m2_possible_cuts = fragmentor.find_bonds(m2)
    if (region is not None and
            set(range(m1.GetNumAtoms())) - (set(region['inside']) | set(region['anchors']))):
        return _region_preserving_crossover(m1, m2, fragmentor, region,
                                            m1_possible_cuts, m2_possible_cuts)

    # get types in common
    m1_cut_types = [ct for _, ct in m1_possible_cuts]
    m2_cut_types = [ct for _, ct in m2_possible_cuts]
    common_cts = list(set(m1_cut_types) & set(m2_cut_types))

    # if no applicable cut types, return None
    if not len(common_cts):
        note_operator('crossover_no_compatible_cuts', primary_cut_count=len(m1_possible_cuts),
                      donor_cut_count=len(m2_possible_cuts))
        logger.debug(f"CrossoverSP: no common cut types: {Chem.MolToSmiles(m1)}, {Chem.MolToSmiles(m2)}")
        return []

    # filter to cuts in common
    m1_possible_cuts = [cut for cut in m1_possible_cuts if cut[1] in common_cts]
    m2_possible_cuts = [cut for cut in m2_possible_cuts if cut[1] in common_cts]

    # shuffle both lists to avoid systematic bias
    random.shuffle(m1_possible_cuts)
    random.shuffle(m2_possible_cuts)

    # get first mol cut
    picked_cut_1 = m1_possible_cuts.pop(0)

    # get second mol cut
    for n, cut in enumerate(m2_possible_cuts):
        if cut[1] == picked_cut_1[1]:
            picked_cut_2 = m2_possible_cuts.pop(n)
            break
    note_operator('crossover_selected_cuts', failure=False,
                  primary_atoms=list(picked_cut_1[0]), primary_types=list(picked_cut_1[1]),
                  donor_atoms=list(picked_cut_2[0]), donor_types=list(picked_cut_2[1]))

    # cut both molecules using BRICS function (turns out this func is quite general!)
    # syntax here is break bond, and assign isotopes as cut types
    m1_pieces = BreakBRICSBonds(m1, [picked_cut_1])
    m2_pieces = BreakBRICSBonds(m2, [picked_cut_2])

    # separate into different mol objects
    m1_pieces = list(Chem.GetMolFrags(m1_pieces, asMols=True))
    m2_pieces = list(Chem.GetMolFrags(m2_pieces, asMols=True))

    # add attachment_idx so that frags can be reconnected (each should only have one)
    m1_pieces = [renumber_frag_attachment_idxs(frag) for frag in m1_pieces]
    m2_pieces = [renumber_frag_attachment_idxs(frag) for frag in m2_pieces]

    # crossover
    # combine 0 with 1, if bond type is wrong, try 1 and 1
    # this could be improved by working out which halves to recombine by afp
    m1_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    m2_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    new_m1 = [m1_pieces[0], m2_pieces[1]]
    new_m2 = [m1_pieces[1], m2_pieces[0]]
    note_operator('crossover_directions', failure=False,
                  first_fragments=_diagnostic_fragments(new_m1),
                  second_fragments=_diagnostic_fragments(new_m2))

    try:
        return _identity_prescreen_children(m1, [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)], region)
    except KeyError:
        note_operator('crossover_asymmetric_connection', exception_type='KeyError')
        logger.debug(f"CrossoverSP: asymmetric bond error: {Chem.MolToSmiles(m1)}, {Chem.MolToSmiles(m2)}")
        # todo does this introduce weirdness to nonasymmetrics? do all four?
        new_m1 = [m1_pieces[1], m2_pieces[1]]
        new_m2 = [m1_pieces[0], m2_pieces[0]]
        note_operator('crossover_alternate_directions', failure=False,
                      first_fragments=_diagnostic_fragments(new_m1),
                      second_fragments=_diagnostic_fragments(new_m2))
        return _identity_prescreen_children(m1, [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)], region)


def mc_operator_factory(operator_name: str):
    """
    Mutation and crossover operator factory

    (`operator_name` must end with '_mutation' or '_crossover' since these have different inputs)
    """
    if operator_name.lower() == "substitute_node_mutation":
        return substitute_node_mutation
    elif operator_name.lower() == "add_node_mutation":
        return add_node_mutation
    elif operator_name.lower() == "delete_node_mutation":
        return delete_node_mutation
    elif operator_name.lower() == "substitute_edge_mutation":
        return substitute_edge_mutation
    elif operator_name.lower() == "single_point_crossover":
        return single_point_crossover
    else:
        raise(Exception(f"operator {operator_name} not recognised"))
