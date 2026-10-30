#!/usr/bin/env python3
"""Run the six BDGen strategies with the shared HA70/0.55 protocol."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HOME=Path(__file__).resolve().parent
RECIPES={
 'property-only':dict(source='core',preset='property_only',objective='property_only',region='none'),
 'bde-only':dict(source='core',preset='bde_only_with_similarity',objective='bde_only_with_similarity',region='none'),
 'joint':dict(source='core',preset='joint',objective='joint',region='none'),
 'random':dict(source='core',preset='random_unique_region',objective='joint',region='random_unique_region'),
 'static':dict(source='core',preset='static_weakest',objective='joint',region='static_weakest'),
 'dynamic':dict(source='core',preset='dynamic_top3_regions',objective='joint',region='dynamic_top3_regions'),
}
PROFILES={
 'manuscript':dict(population_size=200,n_mutations=200,generations=30,number_molecules=100),
 'smoke':dict(population_size=8,n_mutations=8,generations=3,number_molecules=4),
}

def load(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 temp.replace(path)
def check(condition,message):
 if not condition:raise ValueError(message)
def env_for(source,dependency_path=None):
 env=dict(os.environ)
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[key]='1'
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',CUDA_VISIBLE_DEVICES='',BDGEN_DIAGNOSTICS='1',BDGEN_DIAG_SAMPLE_LIMIT='30')
 extra=[str(source)]
 if dependency_path:extra.append(str(Path(dependency_path).expanduser().resolve()))
 if env.get('PYTHONPATH'):extra.append(env['PYTHONPATH'])
 env['PYTHONPATH']=os.pathsep.join(extra)
 return env

def verify_assets(checkpoint,library):
 manifest=load(HOME/'config/assets.json')
 for name,path in [('checkpoint',checkpoint),('fragment_library',library)]:
  check(path.is_file(),f'Missing {name}: {path}')
  check(digest(path)==manifest[name]['sha256'],f'{name} hash differs from the frozen experiment; use the documented model/library.')
 return manifest

def verify_sources():
 manifest=load(HOME/'config/sources.json')
 for version,files in manifest['files'].items():
  root=HOME/version
  for rel,sha in files.items():
   p=root/rel
   check(p.is_file() and digest(p)==sha,f'Frozen source changed or missing: {version}/{rel}')
 return manifest

def command_for(case,strategy,args,output):
 recipe=RECIPES[strategy];source=HOME/recipe['source'];b=PROFILES[args.profile]
 cmd=[sys.executable,str(source/'worker_entry.py'),'--mode','global','--strategy',recipe['preset'],
      '--seed_smiles',case['initial_smiles'],'--smiles_file',str(HOME/'data/initial_molecules.smi'),
      '--fragstore_path',str(args.fragment_library),'--bde_checkpoint',str(args.checkpoint),
      '--device','cpu','--output_dir',str(output),'--population_size',str(b['population_size']),
      '--n_mutations',str(b['n_mutations']),'--generations',str(b['generations']),
      '--number_molecules',str(b['number_molecules']),'--seed','42','--n_jobs','1',
      '--maximize_bde','--keep_properties','--enable_seed_similarity','--min_similarity','0.55',
      '--min-heavy-atoms-ratio','0.70','--selection_strategy','pareto',
      '--structural_filter_profile','druglike_common','--admet_backend','none']
 return cmd,source

def finalize_job(path):
 """Apply the original Pareto selector after exact own-initial exclusion."""
 job=load(path);folder=Path(job['output_dir']);source=HOME/job['recipe']['source']
 sys.path.insert(0,str(source))
 from rdkit import Chem
 module=importlib.import_module('run_bde_frag_gt')
 canon=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True)
 params=load(folder/'run_params.json');pool=load(folder/'summary_all_candidates.json')
 check(params['strategy']==job['recipe']['preset'],'Unexpected strategy preset')
 check(params['objective_mode']==job['recipe']['objective'],'Unexpected scoring objective')
 check(params['target_selection']==job['recipe']['region'],'Unexpected editing region')
 check(params['admet_backend']=='none' and not params['include_admet_in_pareto'],'ADMET must remain post-generation only')
 seed=canon(job['case']['initial_smiles']);eligible=[r for r in pool if canon(r['smiles'])!=seed]
 indices,fronts=module.select_by_pareto(eligible,params['active_objectives'],job['budget']['number_molecules'])
 selected=[eligible[i] for i in indices]
 check(len(selected)==len({canon(r['smiles']) for r in selected}),'Duplicate final candidate structures')
 # A shortfall is reported, never filled with duplicates, alternative seeds, or relaxed filters.
 selection=dict(selection='original per-strategy Pareto selector after exact own-initial exclusion',
                pool_size=len(pool),own_initial_removed=len(pool)-len(eligible),selected=len(selected),
                requested=job['budget']['number_molecules'],shortfall=max(0,job['budget']['number_molecules']-len(selected)),
                selected_indices=indices,fronts=fronts)
 save(folder/'candidates.json',selected);save(folder/'candidate_selection.json',selection)
 trace=load(folder/'generation_trace.json')
 checks=dict(completed_generations=len(trace),requested_generations=job['budget']['generations'],
             generation_children=[t['children'] for t in trace],
             accepted_children=sum(t['children'] for t in trace),
             raw_operator_attempts=sum(t['operator_attempts'] for t in trace),
             successful_search_BDE_evaluation_counter=params['search_debug']['n_bde_eval'],
             seed=42,actual_predictor_forward_calls_not_inferred_from_scorer_counter=True,
             final_unique_candidates=len(selected),selected_initial_excluded=True,
             full_budget_reached=len(trace)==job['budget']['generations'],
             fixed_actual_raw_proposal_budget=False,
             post_generation_ADMET=False,test_only=job['profile']=='smoke',
             effective_unique_scored_children_not_assumed_equal=True)
 save(folder/'budget_audit.json',checks)
 print(json.dumps(dict(status='finalized',case_id=job['case']['case_id'],strategy=job['strategy'],selected=len(selected),checks=checks)))

def run_one(case,strategy,args):
 folder=args.output/case['case_id']/strategy
 job_identity=dict(case=case,strategy=strategy,recipe=RECIPES[strategy],profile=args.profile,
                   budget=PROFILES[args.profile],asset_hashes=args.asset_hashes,seed=42,
                   fixed_HA_ratio='0.70',soft_similarity=0.55,source_manifest_sha256=args.source_manifest_sha256)
 job_id=hashlib.sha256(json.dumps(job_identity,sort_keys=True).encode()).hexdigest()
 if folder.exists():
  old=load(folder/'job.json')
  check(args.resume,f'Output already exists: {folder}; use a new directory or --resume')
  check(old['identity_sha256']==job_id,f'Resume configuration differs: {folder}')
  check(old.get('status')=='completed',f'Existing job is incomplete or failed: {folder}. Inspect it and select a new output directory; it will not be overwritten.')
  check(digest(folder/'candidates.json')==old['candidates_sha256'],'Completed candidate file changed')
  return dict(case_id=case['case_id'],strategy=strategy,status='reused_completed',output_dir=str(folder),identity_sha256=job_id)
 folder.mkdir(parents=True,exist_ok=False)
 cmd,source=command_for(case,strategy,args,folder)
 job=dict(job_identity,identity_sha256=job_id,output_dir=str(folder),status='running',started_utc=now(),command=cmd)
 save(folder/'job.json',job);start=time.monotonic()
 env=env_for(source,args.dependency_path)
 try:
  print(now(),'START',case['case_id'],strategy,flush=True)
  with (folder/'run.log').open('w') as f:
   result=subprocess.run(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT)
  check(result.returncode==0,f'Generation exited with {result.returncode}; inspect {folder}/run.log')
  with (folder/'finalize.log').open('w') as f:
   result=subprocess.run([sys.executable,str(HOME/'run_bdgen.py'),'_finalize',str(folder/'job.json')],cwd=HOME,env=env,stdout=f,stderr=subprocess.STDOUT)
  check(result.returncode==0,f'Final selection failed; inspect {folder}/finalize.log')
  audit=load(folder/'budget_audit.json');job.update(status='completed',finished_utc=now(),elapsed_seconds=time.monotonic()-start,
    candidates_sha256=digest(folder/'candidates.json'),budget_audit=audit)
  save(folder/'job.json',job)
  print(now(),'DONE',case['case_id'],strategy,'candidates',audit['final_unique_candidates'],flush=True)
  return dict(case_id=case['case_id'],strategy=strategy,status='completed',output_dir=str(folder),identity_sha256=job_id,budget_audit=audit)
 except Exception as exc:
  job.update(status='failed',finished_utc=now(),error=str(exc));save(folder/'job.json',job)
  print(now(),'FAILED',case['case_id'],strategy,str(exc),flush=True)
  return dict(case_id=case['case_id'],strategy=strategy,status='failed',output_dir=str(folder),error=str(exc))

# Six distinct public Python call sites; all use the unchanged shared controller.
def run_property_only(case,args):return run_one(case,'property-only',args)
def run_bde_only(case,args):return run_one(case,'bde-only',args)
def run_joint(case,args):return run_one(case,'joint',args)
def run_random(case,args):return run_one(case,'random',args)
def run_static(case,args):return run_one(case,'static',args)
def run_dynamic(case,args):return run_one(case,'dynamic',args)
ROUTES={'property-only':run_property_only,'bde-only':run_bde_only,'joint':run_joint,
        'random':run_random,'static':run_static,'dynamic':run_dynamic}

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('strategy',choices=['all',*RECIPES],help='Run all six strategies or exactly one named strategy')
 p.add_argument('--cases',type=Path,default=HOME/'data/cases.json')
 p.add_argument('--case',action='append',dest='case_ids',help='Frozen source case ID; repeat for several cases')
 p.add_argument('--output',type=Path,default=Path('bdgen_results'))
 p.add_argument('--profile',choices=PROFILES,default='manuscript',help='smoke is explicitly test-only, not a manuscript run')
 p.add_argument('--jobs',type=int,default=1,help='Concurrent independent case/strategy jobs; each job uses one CPU thread')
 p.add_argument('--checkpoint',type=Path,default=HOME/'assets/bde_checkpoint')
 p.add_argument('--fragment-library',type=Path,default=HOME/'assets/chembl_33_chemreps_std_fragstore_brics_filter2.pkl')
 p.add_argument('--dependency-path',type=Path,help='Optional external dependency directory for the frozen remote environment')
 p.add_argument('--plan-only',action='store_true',help='Print the commands without importing the model or launching jobs')
 p.add_argument('--resume',action='store_true',help='Reuse only fully completed jobs with matching configuration and file hashes')
 args=p.parse_args();check(args.jobs>=1,'--jobs must be positive')
 args.output=args.output.expanduser().resolve();args.checkpoint=args.checkpoint.expanduser().resolve();args.fragment_library=args.fragment_library.expanduser().resolve()
 cases=load(args.cases);check(isinstance(cases,list) and cases,'Case file must be a nonempty JSON list')
 for c in cases:check(re.fullmatch(r'[A-Za-z0-9_-]+',c['case_id']) is not None and bool(c['initial_smiles']),'Invalid case ID or SMILES')
 check(len(cases)==len({c['case_id'] for c in cases}),'Duplicate case IDs')
 if args.case_ids:
  check(set(args.case_ids)<=set(c['case_id'] for c in cases),'Unknown --case ID')
  cases=[c for c in cases if c['case_id'] in args.case_ids]
 strategies=list(RECIPES) if args.strategy=='all' else [args.strategy]
 verify_sources();assets=verify_assets(args.checkpoint,args.fragment_library)
 args.asset_hashes={k:assets[k]['sha256'] for k in ['checkpoint','fragment_library']};args.source_manifest_sha256=digest(HOME/'config/sources.json')
 jobs=[(c,s) for c in cases for s in strategies]
 plan=dict(strategy_order=strategies,cases=[c['case_id'] for c in cases],profile=args.profile,
           budget=PROFILES[args.profile],jobs=len(jobs),concurrency=args.jobs,seed=42,
           budget_definition='Same population cap, offspring collection target and generations; actual raw proposals and unique scored children need not match.',
           commands=[command_for(c,s,args,args.output/c['case_id']/s)[0] for c,s in jobs])
 if args.plan_only:print(json.dumps(plan,ensure_ascii=False,indent=2));return
 args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'run_manifest.json'
 if manifest.exists():
  check(args.resume,'Run output exists; choose a new directory or --resume')
  previous=load(manifest)
  for key in ['strategy_order','cases','profile','budget','seed']:
   check(previous['plan'][key]==plan[key],f'Resume plan differs in {key}; choose a new output directory')
  check(previous['asset_hashes']==args.asset_hashes and previous['source_manifest_sha256']==args.source_manifest_sha256,'Resume source or assets differ')
 for c,s in jobs:
  folder=args.output/c['case_id']/s
  if folder.exists():
   check(args.resume and (folder/'job.json').is_file(),f'Existing output needs inspection: {folder}')
   check(load(folder/'job.json').get('status')=='completed',f'Existing job is incomplete or failed: {folder}; use a new output directory')
 state=dict(status='running',started_utc=now(),plan=plan,asset_hashes=args.asset_hashes,source_manifest_sha256=args.source_manifest_sha256,results=[])
 save(manifest,state)
 with ThreadPoolExecutor(max_workers=args.jobs) as pool:
  futures={pool.submit(ROUTES[s],c,args):(c,s) for c,s in jobs}
  for future in as_completed(futures):
   try:result=future.result()
   except Exception as exc:
    c,s=futures[future];result=dict(case_id=c['case_id'],strategy=s,status='failed',error=str(exc))
   state['results'].append(result);state['updated_utc']=now();save(manifest,state)
 state.update(status='completed' if all(r['status']!='failed' for r in state['results']) else 'has_failed_jobs',finished_utc=now());save(manifest,state)
 check(state['status']=='completed','Some jobs failed; retained logs identify the affected jobs')

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='_finalize':finalize_job(Path(sys.argv[2]))
 else:main()
