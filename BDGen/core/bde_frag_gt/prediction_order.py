"""Give BDENET a consistent graph order, then map its bond outputs back."""
import json
import math

from rdkit import Chem


class CanonicalBondAdapter:
    """For BDENET's regenerated 3D geometry, standardize the 2D input first.

    BDENET builds its own conformer rather than evaluating supplied coordinates.
    Equivalent serialized inputs should therefore use the same conformer input.
    Explicit H atoms are retained, and outputs return in the caller's bond order.
    """
    def __init__(self, engine):
        self.engine = engine
        self.cache = {}
        self.engine_calls = 0

    def predict(self, mol):
        reference = Chem.Mol(mol)
        for atom in reference.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(reference, isomericSmiles=True)
        order = json.loads(reference.GetProp("_smilesAtomOutputOrder"))
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        canonical = Chem.MolFromSmiles(smiles, parser)
        if canonical is None or len(order) != mol.GetNumAtoms() or canonical.GetNumAtoms() != len(order):
            return []
        if smiles not in self.cache:
            self.engine_calls += 1
            raw = self.engine.predict(canonical)
            if raw is None:
                return []
            if isinstance(raw, dict):
                if any(i not in raw for i in range(canonical.GetNumBonds())):
                    return []
                values = [float(raw[i]) for i in range(canonical.GetNumBonds())]
            else:
                values = [float(value) for value in raw]
            if len(values) != canonical.GetNumBonds() or not all(math.isfinite(v) for v in values):
                return []
            self.cache[smiles] = values
        values = self.cache[smiles]
        original_to_canonical = {original: new for new, original in enumerate(order)}
        output = []
        for bond in mol.GetBonds():
            mapped = canonical.GetBondBetweenAtoms(original_to_canonical[bond.GetBeginAtomIdx()],
                                                  original_to_canonical[bond.GetEndAtomIdx()])
            if mapped is None or mapped.GetBondType() != bond.GetBondType():
                return []
            output.append(values[mapped.GetIdx()])
        return output
