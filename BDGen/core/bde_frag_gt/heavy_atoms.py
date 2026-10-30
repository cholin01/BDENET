"""A fixed-original-seed molecular-size floor, independent of fitness/regions."""
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING

from rdkit import Chem


@dataclass(frozen=True)
class SeedHeavyAtomConstraint:
    seed_smiles: str
    min_ratio: str
    initial_heavy_atoms: int = field(init=False)
    minimum_heavy_atoms: int = field(init=False)
    ratio_decimal: str = field(init=False)

    def __post_init__(self):
        seed = Chem.MolFromSmiles(self.seed_smiles) if self.seed_smiles else None
        if seed is None or seed.GetNumHeavyAtoms() < 1:
            raise ValueError('A valid original seed with heavy atoms is required')
        try:
            ratio = Decimal(str(self.min_ratio))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError('min-heavy-atoms-ratio must be a finite number in (0,1]') from exc
        if not ratio.is_finite() or not Decimal(0) < ratio <= Decimal(1):
            raise ValueError('min-heavy-atoms-ratio must be a finite number in (0,1]')
        initial = seed.GetNumHeavyAtoms()
        minimum = int((ratio * initial).to_integral_value(rounding=ROUND_CEILING))
        object.__setattr__(self, 'initial_heavy_atoms', initial)
        object.__setattr__(self, 'minimum_heavy_atoms', minimum)
        object.__setattr__(self, 'ratio_decimal', str(ratio))

    def accepts(self, mol):
        return mol is not None and mol.GetNumHeavyAtoms() >= self.minimum_heavy_atoms

    def require(self, mol, stage='molecule'):
        if not self.accepts(mol):
            observed = mol.GetNumHeavyAtoms() if mol is not None else None
            raise ValueError(f'{stage}: heavy atoms {observed} below fixed original-seed minimum '
                             f'{self.minimum_heavy_atoms} (initial={self.initial_heavy_atoms})')

    def metadata(self):
        return dict(enabled=True, baseline='original_initial_molecule',
                    original_seed_smiles=self.seed_smiles,
                    initial_heavy_atoms=self.initial_heavy_atoms,
                    min_heavy_atoms_ratio=float(Decimal(self.ratio_decimal)),
                    min_ratio_decimal=self.ratio_decimal,
                    minimum_heavy_atoms=self.minimum_heavy_atoms,
                    maximum_heavy_atoms=None, hard_filter=True,
                    count_definition='RDKit GetNumHeavyAtoms; atomic number >1; H/D/T and dummy atoms excluded',
                    rounding='ceil(Decimal(str(ratio))*original_seed_heavy_atoms)',
                    baseline_never_rebased_to_parent=True,
                    preserves_size_not_atom_identity_or_scaffold=True)
