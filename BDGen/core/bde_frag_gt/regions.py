"""Fixed seed environments and per-parent dynamic environments, without MCS guesses.

Regions use the existing BRICS decomposition: the fragment containing the target
bond, or the union of its two incident fragments when that bond is a BRICS cut.
External attachment atoms anchor the region. Original atom provenance survives
fragment assembly; donor/library atoms never inherit primary-parent provenance.
"""
from collections import Counter
from rdkit import Chem

SEED = '_bdgen_seed_atom'
PARENT = '_bdgen_parent_atom'
TERMINAL = '_bdgen_static_terminal'
BDE_SCOPE = 'nonring_heavy_single_no_fallback'

def seed_molecule(mol):
    mol=Chem.Mol(mol)
    for atom in mol.GetAtoms():atom.SetIntProp(SEED,atom.GetIdx())
    return mol

def prepare_parent(mol):
    mol=Chem.Mol(mol)
    for atom in mol.GetAtoms():atom.SetIntProp(PARENT,atom.GetIdx())
    return mol

def prepare_donor(mol):
    mol=Chem.Mol(mol)
    for atom in mol.GetAtoms():
        for prop in (SEED,PARENT):
            if atom.HasProp(prop):atom.ClearProp(prop)
    if mol.HasProp(TERMINAL):mol.ClearProp(TERMINAL)
    return mol

def atom_ids(mol,prop=PARENT):
    return {a.GetIntProp(prop) for a in mol.GetAtoms() if a.GetAtomicNum()!=0 and a.HasProp(prop)}

def eligible_bonds(mol):
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBondType()==Chem.BondType.SINGLE and not b.IsInRing()
            and b.GetBeginAtom().GetAtomicNum()>1 and b.GetEndAtom().GetAtomicNum()>1]

def atom_signature(a):
    return (a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsotope(),int(a.GetChiralTag()),a.GetIsAromatic())

def bond_signature(b):
    return (str(b.GetBondType()),int(b.GetStereo()),b.GetIsAromatic())

def graph_edges(mol,ids):
    out={}
    for b in mol.GetBonds():
        u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if u in ids and v in ids:out[tuple(sorted((ids[u],ids[v])))]=bond_signature(b)
    return out

def define_region(mol,bond_index,fragmentor):
    cut_edges={tuple(sorted(pair)) for pair,_ in fragmentor.find_bonds(mol)}
    work=Chem.RWMol(mol)
    for u,v in cut_edges:work.RemoveBond(u,v)
    pieces=[set(x) for x in Chem.GetMolFrags(work.GetMol())]
    bond=mol.GetBondWithIdx(int(bond_index));ends={bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()}
    inside=set().union(*(piece for piece in pieces if piece&ends))
    outside=set(range(mol.GetNumAtoms()))-inside
    anchors={v for u in inside for v in [a.GetIdx() for a in mol.GetAtomWithIdx(u).GetNeighbors()] if v in outside}
    return {'bond_index':int(bond_index),'target_atoms':sorted(ends),'inside':sorted(inside),
            'anchors':sorted(anchors),'whole_molecule':not outside}

def unique_regions(mol,fragmentor):
    """Enumerate eligible edit regions once each, without consulting BDE values."""
    regions=[];seen=set()
    for idx in eligible_bonds(mol):
        region=define_region(mol,idx,fragmentor)
        key=(tuple(region['inside']),tuple(region['anchors']))
        if key in seen:continue
        seen.add(key);regions.append(region)
    return regions

def weakest_regions(mol,values,fragmentor,limit=3):
    """Rank distinct edit regions by their lowest eligible bond prediction."""
    regions=[];seen=set()
    for idx in sorted(eligible_bonds(mol),key=lambda i:(values[i],i)):
        region=define_region(mol,idx,fragmentor)
        key=(tuple(region['inside']),tuple(region['anchors']))
        if key in seen:continue
        seen.add(key);region['min_bde']=float(values[idx]);regions.append(region)
        if len(regions)>=limit:break
    return regions

class RegionPolicy:
    def __init__(self,seed,selection,scorer,fragmentor,random_seed=42):
        import random
        self.selection=selection;self.scorer=scorer;self.fragmentor=fragmentor
        self.random=random.Random(random_seed);self.stats=Counter();self.seed=seed_molecule(seed)
        self.definition=None
        if selection=='static_weakest':
            values=scorer._get_bde_values(self.seed,force=True)
            if not values:raise ValueError('Cannot define initial weakest environment: BDE unavailable')
            eligible=eligible_bonds(self.seed)
            if not eligible:raise ValueError('No eligible initial bond')
            idx=min(eligible,key=lambda i:values[i]);self.definition=define_region(self.seed,idx,fragmentor)
            inside=set(self.definition['inside']);self.outside=set(range(seed.GetNumAtoms()))-inside
            self.anchor_ids=set(self.definition['anchors'])
            self.outside_atoms={i:atom_signature(self.seed.GetAtomWithIdx(i)) for i in self.outside}
            self.outside_edges=graph_edges(self.seed,{i:i for i in self.outside})
            self.definition.update(initial_bde=float(values[idx]),outside_seed_ids=sorted(self.outside),
                                   boundary_seed_ids=sorted(self.anchor_ids),initial_bond_type=str(self.seed.GetBondWithIdx(idx).GetBondType()))

    def locate_static(self,mol):
        if mol.HasProp(TERMINAL) and mol.GetBoolProp(TERMINAL):return None
        identity={}
        for atom in mol.GetAtoms():
            if atom.HasProp(SEED):
                i=atom.GetIntProp(SEED)
                if i in identity:return None
                identity[i]=atom.GetIdx()
        if not self.outside.issubset(identity):return None
        if any(atom_signature(mol.GetAtomWithIdx(identity[i]))!=sig for i,sig in self.outside_atoms.items()):return None
        mapped={identity[i]:i for i in self.outside}
        if graph_edges(mol,mapped)!=self.outside_edges:return None
        outside=set(mapped);inside=set(range(mol.GetNumAtoms()))-outside
        if not inside:return None
        anchors={identity[i] for i in self.anchor_ids}
        if any(not any(n.GetIdx() in inside for n in mol.GetAtomWithIdx(a).GetNeighbors()) for a in anchors):return None
        # Require one identifiable edited region, rather than guessing among fragments.
        pending=[next(iter(inside))];seen=set()
        while pending:
            i=pending.pop()
            if i in seen:continue
            seen.add(i);pending.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in inside-seen)
        if seen!=inside:return None
        return {'inside':sorted(inside),'anchors':sorted(anchors),'whole_molecule':not outside,
                'source':'fixed_initial_weakest_environment','boundary_seed_ids':sorted(self.anchor_ids)}

    def target(self,mol):
        self.stats['requests']+=1
        if self.selection=='none':return None
        if self.selection=='static_weakest':
            region=self.locate_static(mol)
            if region is not None:region['initial_target_seed_ids']=self.definition['target_atoms']
        else:
            eligible=eligible_bonds(mol)
            if not eligible:region=None
            elif self.selection=='random':
                region=define_region(mol,self.random.choice(eligible),self.fragmentor)
            elif self.selection=='random_unique_region':
                candidates=unique_regions(mol,self.fragmentor)
                selected=self.random.randrange(len(candidates))
                region=dict(candidates[selected],candidate_region_count=len(candidates),
                            selected_region_index=selected)
            else:
                values=self.scorer._get_bde_values(mol,force=True)
                if values and self.selection in ('dynamic_top3_regions','dynamic_top5_regions'):
                    limit=5 if self.selection=='dynamic_top5_regions' else 3
                    candidates=weakest_regions(mol,values,self.fragmentor,limit=limit)
                    selected=self.random.randrange(len(candidates))
                    region=dict(candidates[selected],candidate_regions=candidates,selected_region_rank=selected+1)
                else:
                    region=define_region(mol,min(eligible,key=lambda i:values[i]),self.fragmentor) if values else None
            if region is not None:region['source']=self.selection
            if region is not None:
                region['target_seed_ids']=[mol.GetAtomWithIdx(i).GetIntProp(SEED) if mol.GetAtomWithIdx(i).HasProp(SEED) else None for i in region['target_atoms']]
        self.stats['resolved' if region is not None else 'unresolved']+=1
        return region

    def validate(self,parent,child,region):
        ids={a.GetIdx():a.GetIntProp(PARENT) for a in child.GetAtoms() if a.HasProp(PARENT)}
        counts=Counter(ids.values())
        if any(n!=1 for n in counts.values()):return {'accepted':False,'reason':'duplicate_parent_identity'}
        survivors={v:k for k,v in ids.items()}
        deleted=set(range(parent.GetNumAtoms()))-set(survivors)
        changed={i for i,j in survivors.items() if atom_signature(parent.GetAtomWithIdx(i))!=atom_signature(child.GetAtomWithIdx(j))}
        before=graph_edges(parent,{i:i for i in range(parent.GetNumAtoms())});after=graph_edges(child,ids)
        edited=set(deleted)|changed
        for edge in before.keys()|after.keys():
            if before.get(edge)!=after.get(edge):edited.update(edge)
        for bond in child.GetBonds():
            u,v=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            if u in ids and v not in ids:edited.add(ids[u])
            if v in ids and u not in ids:edited.add(ids[v])
        if Chem.MolToSmiles(parent,isomericSmiles=True)==Chem.MolToSmiles(child,isomericSmiles=True):
            return {'accepted':False,'reason':'same_structure'}
        allowed=set(region['inside'])|set(region['anchors'])
        if not edited or not edited.issubset(allowed):
            return {'accepted':False,'reason':'outside_target_environment','affected_parent_atoms':sorted(edited)}
        terminal=False
        if self.selection=='static_weakest':
            terminal=self.locate_static(child) is None
            child.SetBoolProp(TERMINAL,terminal)
        target_survives=None
        if self.selection=='static_weakest':
            child_seed={a.GetIntProp(SEED):a.GetIdx() for a in child.GetAtoms() if a.HasProp(SEED)}
            u,v=self.definition['target_atoms']
            bond=child.GetBondBetweenAtoms(child_seed[u],child_seed[v]) if u in child_seed and v in child_seed else None
            target_survives=bond is not None and str(bond.GetBondType())==self.definition['initial_bond_type']
        return {'accepted':True,'reason':'accepted','affected_parent_atoms':sorted(edited),
                'deleted_parent_atoms':sorted(deleted),'added_atoms':child.GetNumAtoms()-len(ids),
                'static_terminal':terminal,'initial_target_bond_survives':target_survives,
                'boundary_preserved':not terminal if self.selection=='static_weakest' else None}

def choose_fragment(frags,region,candidates=None):
    import numpy as np
    choices=list(range(len(frags))) if candidates is None else list(candidates)
    if region is not None:
        inside=set(region['inside'])
        choices=[i for i in choices if atom_ids(frags[i]) and atom_ids(frags[i]).issubset(inside)]
    return int(np.random.choice(choices)) if choices else None

def restrict_cuts(cuts,region):
    if region is None:return cuts
    inside=set(region['inside'])
    return [cut for cut in cuts if set(cut[0])&inside]
