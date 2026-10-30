"""Passive, bounded diagnostics for the unchanged molecular search.

No predictor or random-number generator is used here.  RDKit writers and graph
inspection operate on private molecule copies, never on search objects.  A
provenance-based observation is not evidence of a chemical mapping error.
"""
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import json
import math
from numbers import Integral, Real
from pathlib import Path

from rdkit import Chem


PARENT = '_bdgen_parent_atom'
SEED = '_bdgen_seed_atom'
_ACTIVE = ContextVar('bdgen_diagnostic_recorder', default=None)


def _passive(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self._error(method.__name__, exc)
            return None
    return guarded


def _safe(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_safe(item) for item in value), key=str)
    if isinstance(value, Chem.Mol):
        return _mol_record(value, 'detail_molecule')
    # Do not invoke arbitrary repr/serialization methods on search objects.
    return {'unsupported_type': type(value).__name__}


def _mol_record(mol, role):
    if mol is None:
        return None
    record = {'role': role, 'smiles': None, 'molblock': None,
              'atom_provenance': [], 'serialization_errors': []}
    try:
        copied = Chem.Mol(mol)
    except Exception as exc:
        record['serialization_errors'].append({'operation': 'copy',
                                                'type': type(exc).__name__, 'message': str(exc)})
        return record
    for atom in copied.GetAtoms():
        row = {'atom_index': atom.GetIdx(), 'atomic_number': atom.GetAtomicNum(),
               'formal_charge': atom.GetFormalCharge(), 'isotope': atom.GetIsotope(),
               'chiral_tag': int(atom.GetChiralTag()), 'aromatic': atom.GetIsAromatic(),
               'parent_id': None, 'seed_id': None}
        for prop, key in ((PARENT, 'parent_id'), (SEED, 'seed_id')):
            if atom.HasProp(prop):
                try:
                    row[key] = atom.GetIntProp(prop)
                except Exception as exc:
                    row[key + '_error'] = type(exc).__name__
        record['atom_provenance'].append(row)
    for operation, writer in (
            ('smiles', lambda item: Chem.MolToSmiles(item, isomericSmiles=True)),
            ('molblock', lambda item: Chem.MolToMolBlock(item, includeStereo=True, kekulize=False))):
        try:
            record[operation] = writer(Chem.Mol(copied))
        except Exception as exc:
            record['serialization_errors'].append({'operation': operation,
                                                    'type': type(exc).__name__, 'message': str(exc)})
    return record


def _atom_signature(atom):
    return (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
            int(atom.GetChiralTag()), atom.GetIsAromatic())


def _edges(mol, ids):
    result = {}
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in ids and v in ids:
            result[tuple(sorted((ids[u], ids[v])))] = (
                str(bond.GetBondType()), int(bond.GetStereo()), bond.GetIsAromatic())
    return result


def _region_analysis(parent, child, region):
    """Mirror the validator's footprint, while exposing uncertain identities."""
    parent, child = Chem.Mol(parent), Chem.Mol(child)
    n_parent = parent.GetNumAtoms()
    allowed = set(region.get('inside', ())) | set(region.get('anchors', ()))
    protected = set(range(n_parent)) - allowed
    ids, invalid, unmarked = {}, [], []
    for atom in child.GetAtoms():
        if not atom.HasProp(PARENT):
            unmarked.append(atom.GetIdx())
            continue
        try:
            identity = atom.GetIntProp(PARENT)
        except Exception:
            invalid.append({'child_atom_index': atom.GetIdx(), 'reason': 'non_integer_parent_marker'})
            continue
        if not 0 <= identity < n_parent:
            invalid.append({'child_atom_index': atom.GetIdx(), 'parent_id': identity,
                            'reason': 'parent_marker_out_of_range'})
            continue
        ids[atom.GetIdx()] = identity
    counts = Counter(ids.values())
    duplicate_ids = sorted(identity for identity, count in counts.items() if count > 1)
    survivors = {identity: idx for idx, identity in ids.items()}
    deleted = set(range(n_parent)) - set(survivors)
    changed = {identity for identity, idx in survivors.items()
               if _atom_signature(parent.GetAtomWithIdx(identity)) !=
               _atom_signature(child.GetAtomWithIdx(idx))}
    before = _edges(parent, {i: i for i in range(n_parent)})
    after = _edges(child, ids)
    affected = deleted | changed
    edge_changes = []
    for edge in sorted(before.keys() | after.keys()):
        if before.get(edge) != after.get(edge):
            affected.update(edge)
            edge_changes.append({'parent_atom_indices': list(edge),
                                 'before': _safe(before.get(edge)), 'after': _safe(after.get(edge))})
    new_connections, new_connection_atoms = [], set()
    for bond in child.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (u in ids) != (v in ids):
            tracked, other = (u, v) if u in ids else (v, u)
            new_connection_atoms.add(ids[tracked])
            new_connections.append({'parent_atom_index': ids[tracked],
                                    'tracked_child_atom_index': tracked,
                                    'untracked_child_atom_index': other})
    affected |= new_connection_atoms
    flags = []
    if duplicate_ids:
        flags.append('duplicate_parent_markers')
    flags.extend(sorted({item['reason'] for item in invalid}))
    if deleted & protected:
        flags.append('unrepresented_protected_parent_atoms')
    if not ids and protected:
        flags.append('no_parent_markers_with_protected_atoms')
    parent_marker_mismatch = []
    for atom in parent.GetAtoms():
        if atom.HasProp(PARENT):
            try:
                if atom.GetIntProp(PARENT) != atom.GetIdx():
                    parent_marker_mismatch.append(atom.GetIdx())
            except Exception:
                parent_marker_mismatch.append(atom.GetIdx())
    if parent_marker_mismatch:
        flags.append('primary_parent_marker_index_mismatch')
    return {'parent_atom_count': n_parent, 'child_atom_count': child.GetNumAtoms(),
            'allowed_parent_atoms': sorted(allowed), 'protected_parent_atoms': sorted(protected),
            'tracked_parent_atoms': sorted(set(ids.values())),
            'untracked_child_atoms': unmarked, 'duplicate_parent_ids': duplicate_ids,
            'invalid_parent_markers': invalid,
            'deleted_or_unrepresented_parent_atoms': sorted(deleted),
            'changed_parent_atoms': sorted(changed), 'edge_changes': edge_changes,
            'new_connections': new_connections,
            'new_connection_parent_atoms': sorted(new_connection_atoms),
            'affected_parent_atoms': sorted(affected),
            'outside_parent_atoms': sorted(affected - allowed),
            'marker_suspicion_flags': flags,
            'interpretation': 'Missing provenance may reflect deletion or reconstruction; it does not establish chemical change or a validator false positive.'}


def _stats():
    return {'events': {}, 'reasons': {}, 'per_operator': {}}


def _increment(stats, event, n, reason, operator):
    stats['events'][event] = stats['events'].get(event, 0) + n
    if reason is not None:
        by_reason = stats['reasons'].setdefault(event, {})
        by_reason[reason] = by_reason.get(reason, 0) + n
    by_operator = stats['per_operator'].setdefault(operator, {'events': {}, 'reasons': {}})
    by_operator['events'][event] = by_operator['events'].get(event, 0) + n
    if reason is not None:
        reasons = by_operator['reasons'].setdefault(event, {})
        reasons[reason] = reasons.get(reason, 0) + n


class DiagnosticRecorder:
    def __init__(self, output_dir, sample_limit=30, metadata=None):
        self.output_dir = Path(output_dir)
        self.sample_limit = max(0, int(sample_limit))
        self.metadata = _safe(metadata or {})
        self.cumulative = _stats()
        self.per_generation = []
        self.samples_written = {}
        self.logging_errors = []
        self.logging_error_count = 0
        self.generation = None
        self._current = None
        self._attempt = None
        self._operator = None
        self._parent = self._donor = self._region = None
        self._notes = []
        self._outcome_recorded = False
        self._token = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._error('mkdir', exc)

    def _error(self, operation, exc):
        try:
            self.logging_error_count += 1
            if len(self.logging_errors) < 100:
                self.logging_errors.append({'operation': operation, 'type': type(exc).__name__,
                                            'message': str(exc), 'generation': self.generation,
                                            'attempt': self._attempt, 'operator': self._operator})
        except Exception:
            pass

    @_passive
    def start_generation(self, generation):
        if self._current is not None:
            self.finish_generation()
        self.end_attempt()
        self.generation = int(generation)
        self._current = dict(generation=self.generation, **_stats())

    @_passive
    def start_attempt(self, attempt, operator, parent, region, donor=None):
        self.end_attempt()
        self._attempt, self._operator = int(attempt), str(operator)
        self._parent = Chem.Mol(parent) if parent is not None else None
        self._donor = Chem.Mol(donor) if donor is not None else None
        self._region, self._notes = _safe(region), []
        self._outcome_recorded = False
        self._token = _ACTIVE.set(self)

    @_passive
    def count(self, event, n=1, reason=None):
        event, n = str(event), int(n)
        reason = str(reason) if reason is not None else None
        operator = self._operator or '__generation__'
        _increment(self.cumulative, event, n, reason, operator)
        if self._current is not None:
            _increment(self._current, event, n, reason, operator)

    def _append(self, filename, row):
        line = json.dumps(_safe(row), ensure_ascii=False, allow_nan=False, sort_keys=True)
        with (self.output_dir / filename).open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    @_passive
    def sample(self, reason, child=None, details=None):
        reason, operator = str(reason), self._operator or '__generation__'
        written = self.samples_written.setdefault(operator, {})
        if written.get(reason, 0) >= self.sample_limit:
            return
        details = _safe(details or {})
        row = {'schema_version': 1, 'metadata': self.metadata,
               'generation': self.generation, 'attempt': self._attempt,
               'operator': operator, 'reason': reason,
               'stage': details.get('stage', 'population'),
               'parent': _mol_record(self._parent, 'primary_parent_at_attempt_start'),
               'donor': _mol_record(self._donor, 'source_donor_before_prepare'),
               'child': _mol_record(child, 'candidate'),
               'region': self._region, 'attempt_notes': list(self._notes), 'details': details}
        for key in ('case_id', 'seed_id', 'strategy', 'run_id'):
            if key in self.metadata:
                row[key] = self.metadata[key]
        self._append('diagnostic_samples.jsonl', row)
        written[reason] = written.get(reason, 0) + 1

    @_passive
    def operator_outcome(self, produced_children, exception_type=None):
        """Count each attempt once, regardless of how many internal retries fail.

        Reason breakdowns identify only the last observed failure note in that
        attempt.  They do not establish why an operator ultimately returned no
        child.  Exceptions retain the population's separate primary accounting.
        """
        if self._attempt is None or self._outcome_recorded:
            return
        self._outcome_recorded = True
        failures = [note for note in self._notes if note.get('failure', True)]
        last_reason = failures[-1]['reason'] if failures else None
        construction_reasons = {'construction_failure', 'edge_construction_rejected',
                                'crossover_asymmetric_connection'}
        construction = any(note['reason'] in construction_reasons for note in failures)
        fragmentation = any(note['reason'] == 'fragmentation_failure' for note in failures)
        if construction:
            self.count('operator_attempts_with_construction_failure', reason=last_reason)
        if fragmentation:
            self.count('operator_attempts_with_fragmentation_failure', reason=last_reason)
        if int(produced_children) > 0:
            if failures:
                self.count('operator_success_after_internal_failure', reason=last_reason)
        elif exception_type is None:
            if construction:
                self.count('operator_empty_due_to_construction_failure', reason=last_reason)
            elif failures:
                self.count('operator_no_feasible_result_attempts', reason=last_reason)
            else:
                # Nonfailure observation notes do not explain an empty return.
                self.count('operator_empty_without_internal_reason')

    @_passive
    def region_result(self, child, report):
        original_report = _safe(report)
        analysis = None
        try:
            analysis = _region_analysis(self._parent, child, self._region or {})
        except Exception as exc:
            analysis = {'diagnostic_error': {'type': type(exc).__name__, 'message': str(exc)}}
            self._error('region_analysis', exc)
        self.count('region_checked_children')
        for flag in analysis.get('marker_suspicion_flags', ()):
            self.count('region_marker_observations', reason=flag)
        for key in ('deleted_or_unrepresented_parent_atoms', 'changed_parent_atoms',
                    'new_connection_parent_atoms', 'affected_parent_atoms', 'outside_parent_atoms'):
            if analysis.get(key):
                self.count('region_footprint_atom_observations', n=len(analysis[key]), reason=key)
        if analysis.get('edge_changes'):
            self.count('region_edge_change_observations', n=len(analysis['edge_changes']))
        if report.get('accepted'):
            self.count('region_accepted_children', reason=str(report.get('reason', 'accepted')))
            return
        reason = str(report.get('reason', 'unknown_region_rejection'))
        if reason == 'outside_target_environment' and 'affected_parent_atoms' in analysis:
            if not analysis['affected_parent_atoms']:
                reason = 'no_tracked_parent_edit'
            elif analysis['outside_parent_atoms']:
                reason = 'tracked_outside_target_environment'
            else:
                reason = 'outside_label_not_reproduced'
        self.count('region_rejected_children', reason=reason)
        protected = set(analysis.get('protected_parent_atoms', ()))
        for label, key in (('unrepresented_outside', 'deleted_or_unrepresented_parent_atoms'),
                           ('changed_outside', 'changed_parent_atoms'),
                           ('new_connection_outside', 'new_connection_parent_atoms')):
            if protected & set(analysis.get(key, ())):
                self.count('region_rejection_causes', reason=label)
        if any(protected & set(change['parent_atom_indices'])
               for change in analysis.get('edge_changes', ())):
            self.count('region_rejection_causes', reason='edge_change_outside')
        self.sample(reason, child=child,
                    details={'stage': 'region_validation', 'original_report': original_report,
                             'diagnostic_analysis': analysis})

    @_passive
    def end_attempt(self):
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        self._attempt = self._operator = None
        self._parent = self._donor = self._region = None
        self._notes = []
        self._outcome_recorded = False

    @_passive
    def finish_generation(self):
        self.end_attempt()
        if self._current is None:
            return
        current = _safe(self._current)
        self.per_generation.append(current)
        # Close the in-memory generation before I/O: a failed JSONL write must
        # neither append this generation again on the next start nor prevent the
        # independent atomic cumulative snapshot from recording the error.
        self._current = None
        try:
            self._append('diagnostic_generation_counts.jsonl',
                         dict(schema_version=1, metadata=self.metadata, **current))
        except Exception as exc:
            self._error('generation_jsonl', exc)
        snapshot = {'schema_version': 1, 'metadata': self.metadata,
                    'sample_limit_per_operator_reason': self.sample_limit,
                    'completed_generations': len(self.per_generation),
                    'cumulative': self.cumulative, 'per_generation': self.per_generation,
                    'per_operator': self.cumulative['per_operator'],
                    'samples_written': self.samples_written,
                    'operator_outcome_reason_semantics':
                        'Last observed internal failure note only; not a causal attribution of the final outcome.',
                    'logging_error_count': self.logging_error_count,
                    'logging_errors': self.logging_errors}
        path = self.output_dir / 'diagnostic_counts.json'
        temporary = self.output_dir / 'diagnostic_counts.json.tmp'
        temporary.write_text(json.dumps(_safe(snapshot), ensure_ascii=False, indent=2,
                                        allow_nan=False, sort_keys=True) + '\n', encoding='utf-8')
        temporary.replace(path)


def note_operator(reason, **details):
    """Observe an internal branch, including failures followed by a later retry."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    try:
        failure = bool(details.pop('failure', True))
        note = {'reason': str(reason), 'failure': failure, 'details': _safe(details)}
        recorder._notes.append(note)
        recorder.count('operator_internal', reason=reason)
        recorder.count('operator_internal_failure_notes' if failure else
                       'operator_internal_observations', reason=reason)
        if failure:
            recorder.sample(reason, details={'stage': 'operator_internal',
                                              'failure': True, 'operator_details': details,
                                              'interpretation': 'This internal branch can fail before a subsequent retry succeeds.'})
    except Exception as exc:
        recorder._error('note_operator', exc)
