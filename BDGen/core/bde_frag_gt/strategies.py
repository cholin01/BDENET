"""Objective ablation layered on the public scorer; original min/mean terms retained."""
import math
from rdkit import Chem
from bde_frag_gt.optimizer import BDEPropertyScorer,_bounded_score,_target_score
from bde_frag_gt.regions import eligible_bonds

PRESETS={'property_only':('property_only','none'),'bde_only':('bde_only','none'),
         'bde_only_with_similarity':('bde_only_with_similarity','none'),
         'joint':('joint','none'),'random_bond':('joint','random'),
         'static_weakest':('joint','static_weakest'),'dynamic_rescan':('joint','dynamic_weakest'),
         'dynamic_top3_regions':('joint','dynamic_top3_regions'),
         'dynamic_top5_regions':('joint','dynamic_top5_regions'),
         'random_unique_region':('joint','random_unique_region'),
         'random_unique_region_property_only':('property_only','random_unique_region')}

class StrategyScorer(BDEPropertyScorer):
    def __init__(self,predictor,config,strategy):
        super().__init__(predictor,config);self.strategy=strategy;self.objective,self.selection=PRESETS[strategy]
        keys=list(BDEPropertyScorer.pareto_objective_keys)
        if not config.seed_smiles:keys.remove('obj_similarity')
        self.pareto_objective_keys=[k for k in keys if (self.objective!='property_only' or k not in ['obj_min_bde','obj_mean_bde'])
                                    and (self.objective!='bde_only' or k in ['obj_min_bde','obj_mean_bde'])]
        if self.objective=='bde_only_with_similarity':
            if self._seed_fp is None or self._seed_fp.GetNumOnBits()==0:
                raise ValueError('bde_only_with_similarity requires a valid initial seed fingerprint; enable --enable_seed_similarity')
            self.pareto_objective_keys=['obj_min_bde','obj_mean_bde','obj_similarity']
        config.use_two_stage_bde=False

    def _merge_values(self,mol,cheap_values,bdes):
        # Scoring and target selection share precisely the same bond scope.
        eligible=eligible_bonds(mol)
        scoped=[float(bdes[i]) for i in eligible] if bdes else []
        values=dict(cheap_values,bde_scope_count=len(eligible))
        # Zero is an internal placeholder; has_bde=0 prevents BDE selection,
        # and the report writer converts missing physical values to JSON null.
        values.update(min_bde=min(scoped) if scoped else 0.0,
                      mean_bde=sum(scoped)/len(scoped) if scoped else 0.0,
                      has_bde=float(bool(scoped)))
        return values

    def _get_bde_values(self,mol,force=False,cheap_values=None,cheap_score=None):
        # Indexed predictions must never be shared by canonical SMILES alone.
        key=(Chem.MolToSmiles(mol,isomericSmiles=True),
             tuple((a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsotope(),int(a.GetChiralTag())) for a in mol.GetAtoms()),
             tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType())) for b in mol.GetBonds()))
        if key in self._bde_cache:return self._bde_cache[key]
        try:
            values=self.predictor.predict_bond_bde(mol)
            if len(values)!=mol.GetNumBonds() or not values or not all(math.isfinite(float(v)) for v in values):raise ValueError('Incomplete BDE')
            self._bde_cache[key]=list(map(float,values));self.n_bde_eval+=1
        except Exception:
            self._bde_cache[key]=None;self.n_bde_failed+=1
        return self._bde_cache[key]

    def _basic_values(self,mol,force_bde=False):
        if self.objective=='bde_only_with_similarity':
            # Descriptors remain available for reporting, but neither their
            # property subscore nor its property penalties enter this arm.
            cheap=self._cheap_values(mol)
            if cheap is None:return None
            bdes=self._get_bde_values(mol,force=True,cheap_values=cheap)
            return self._merge_values(mol,cheap,bdes)
        if self.objective=='property_only' and not force_bde:
            cheap=self._cheap_values(mol)
            return self._merge_values(mol,cheap,None) if cheap is not None else None
        return super()._basic_values(mol,force_bde=True)

    def _score_from_values(self,v,mol):
        if self.objective=='property_only':return self._cheap_score_from_values(v)
        if v.get('has_bde',0)<.5:return -1.0
        if self.objective=='joint':return super()._score_from_values(v,mol)
        c=self.config
        if c.weakest_bde_range is not None:a=_bounded_score(v['min_bde'],*c.weakest_bde_range)
        elif c.maximize_bde:a=min(1,v['min_bde']/max(1,c.target_min_bde))
        else:a=_target_score(v['min_bde'],c.target_min_bde,50)
        b=min(1,v['mean_bde']/max(1,c.target_mean_bde)) if c.maximize_bde else _target_score(v['mean_bde'],c.target_mean_bde,60)
        if self.objective=='bde_only_with_similarity':
            similarity=float(v['similarity'])
            penalty=min(0.7,2.0*max(0.0,c.min_similarity-similarity))
            adjusted_similarity=max(0.0,min(1.0,similarity-penalty))
            return (c.weight_bde_min*a+c.weight_bde_mean*b+c.weight_similarity*adjusted_similarity)/(c.weight_bde_min+c.weight_bde_mean+c.weight_similarity)
        return (c.weight_bde_min*a+c.weight_bde_mean*b)/(c.weight_bde_min+c.weight_bde_mean)

    def _objectives_from_values(self,v):
        if self.objective=='bde_only_with_similarity':
            return {'obj_min_bde':float(v['min_bde']),'obj_mean_bde':float(v['mean_bde']),
                    'obj_similarity':float(v['similarity'])}
        return super()._objectives_from_values(v)

    def score_mols(self,mols):
        result=[]
        for mol in mols:
            if mol is None:result.append(-1.0);continue
            v=self._basic_values(mol)
            result.append(self._score_from_values(v,mol) if v is not None else -1.0)
        return result

    def score_list(self,smiles):return self.score_mols([Chem.MolFromSmiles(s) for s in smiles])

    def objectives(self,mol):
        mol=Chem.MolFromSmiles(mol) if isinstance(mol,str) else mol
        if mol is None:return None
        v=self._basic_values(mol)
        if v is None or (self.objective!='property_only' and v.get('has_bde',0)<.5):return None
        return self._objectives_from_values(v)

    def metrics(self,mol):return self._basic_values(mol,force_bde=True)
