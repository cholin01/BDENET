# BDGen

BDGen combines FragGT fragment-based molecular editing with BDENET bond dissociation energy (BDE) predictions and molecular property objectives. This distribution provides a single command-line entry point for six optimization strategies, together with the model checkpoint, BRICS fragment library, and starting structures for 15 lead-optimization cases.

The strategies share a common implementation and use the same initial structures, fragment library, model checkpoint, structural checks, heavy-atom constraint, and similarity guidance.

The checkpoint contains only the model parameter state required for inference. Network configuration and output normalization constants are defined in the inference code; optimizer state, scheduler state, and training-progress records are excluded.

## Installation

The target environment is **Linux x86_64 with Python 3.12**. The commands below install PyTorch 2.6.0 and matching CPU extensions for PyTorch Geometric. Run them from the package directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install --only-binary=:all: --no-index --no-deps \
  torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

python -m pip install -r requirements.txt
```

## Optimization strategies

| Command | Optimization objective | Editing-region selection |
|---|---|---|
| `property-only` | Molecular properties and initial-structure similarity | Standard FragGT editing without weak-region targeting |
| `bde-only` | Minimum and mean BDE, with initial-structure similarity | Standard FragGT editing without weak-region targeting |
| `joint` | BDE and molecular properties, with initial-structure similarity | Standard FragGT editing without weak-region targeting |
| `random` | Same joint objective as `joint` | Uniform sampling from distinct eligible regions of the current parent |
| `static` | Same joint objective as `joint` | Tracking of the region defined by the initially weakest eligible bond |
| `dynamic` | Same joint objective as `joint` | Uniform sampling from up to three distinct low-BDE regions of the current parent |

The first three strategies compare optimization objectives. The last three compare editing-region selection under the same joint objective.

## Shared protocol

The 15 starting cases are numbered consecutively from MO-001 to MO-015.

| Setting | Default value or definition |
|---|---|
| Starting structures | 15 cases in `data/cases.json` |
| Heavy-atom constraint | At least `ceil(0.70 × initial heavy-atom count)` |
| Similarity guidance | Soft reference value of 0.55 relative to the original starting molecule |
| Similarity measure | Morgan fingerprints, radius 2, 2,048 bits, without chirality; Tanimoto similarity |
| Eligible BDE bonds | Non-ring bonds between two heavy atoms with RDKit bond type `SINGLE` |
| BDE objectives | Minimum and arithmetic mean over the same eligible bond set |
| Generation MW ceiling | 1,500 Da |
| Population cap | 200 |
| Offspring collection target | 200 per generation |
| Generations | 30 |
| Random seed | 42 |
| Final candidate cap | 100 per case and strategy |

Property descriptors include QED, synthetic accessibility, molecular weight, cLogP, and TPSA, with their scoring rules defined in `core/bde_frag_gt/`.

The configured population cap, offspring collection target, generation count, and final candidate cap are shared across strategies.

Final candidates are selected from the retained population after excluding structures identical to the corresponding starting molecule, using canonical isomeric SMILES. Nondominated fronts are included in sequence, with crowding-distance truncation when necessary.

## Usage

### Functional check

Run a short workflow for all six strategies on one case:

```bash
python run_bdgen.py all --case MO-008 --profile smoke \
  --output smoke_results --jobs 2
```

The `smoke` profile uses a population cap of 8, an offspring target of 8, 3 generations, and a final candidate cap of 4. It is intended for installation and execution checks.

### Full benchmark

Run all 15 cases under all six strategies, producing 90 independent case–strategy jobs:

```bash
python run_bdgen.py all --output results --jobs 2
```

Run a single strategy or selected cases:

```bash
python run_bdgen.py dynamic --output dynamic_results

python run_bdgen.py random --case MO-006 --case MO-008 \
  --output selected_random_results --jobs 2
```

### Command-line options

| Option | Purpose |
|---|---|
| `--case ID` | Restrict execution to a case; repeat to select multiple cases |
| `--cases PATH` | Read a case list containing `case_id` and `initial_smiles` fields |
| `--output PATH` | Set the output directory |
| `--profile manuscript` | Use the full protocol; this is the default |
| `--profile smoke` | Use the short functional-check profile |
| `--jobs N` | Run up to N independent jobs concurrently; each job uses one CPU thread |
| `--plan-only` | Print the planned commands without loading the model or launching jobs |
| `--resume` | Reuse completed jobs with matching configuration and candidate-file hashes |
| `--checkpoint PATH` | Specify another location for the same verified checkpoint |
| `--fragment-library PATH` | Specify another location for the same verified fragment library |
| `--dependency-path PATH` | Add an external Python dependency directory to worker environments |

## Outputs

Outputs are written under `<output>/<case_id>/<strategy>/`. The principal files are:

| File | Contents |
|---|---|
| `candidates.json` | Final candidates after initial-structure exclusion and Pareto selection; the authoritative candidate output |
| `candidate_selection.json` | Final selection indices, fronts, counts, and any shortfall |
| `summary_all_candidates.json` | Candidate records for the final retained population |
| `run_params.json` | Effective optimization parameters, objectives, and region settings |
| `generation_trace.json` | Per-generation offspring counts and operator-attempt statistics |
| `edit_audit.jsonl` | Editing-region and edit-validation records |
| `budget_audit.json` | Realized search-budget counts and completion information |
| `job.json` | Job configuration, status, timing, and candidate-file hash |
| `run.log` / `finalize.log` | Optimization and final-selection logs |

## ADMET evaluation

The six-strategy entry point explicitly disables ADMET prediction and excludes ADMET endpoints from generation, filtering, and Pareto selection. The property-evaluation module retains an optional ADMET-AI adapter for separate post-generation evaluation.
