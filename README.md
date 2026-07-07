# 🧬 BDENET：Bond Energetics as a Physical Coordinate for Molecular Discovery and Design

**BDENET** is a bond-level deep learning framework for **bond dissociation energy (BDE) prediction** and **BDE-aware molecular modeling**.

The repository provides a core BDE prediction model, a BDE-guided molecular optimization module, and several downstream BDE-aware molecular modeling tools.



<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/1aa5b06b-28c6-4041-a31e-bedf6ce1dd27" 
    alt="BDENET overview" 
    width="100%" 
  />
</p>


---

## ✨ Overview

BDENET focuses on modeling molecular bonds from the perspective of energetic stability.

The repository contains three main parts:

| Module              | Path           | Description                                               |
| ------------------- | -------------- | --------------------------------------------------------- |
| 🧠 **BDENET**       | `BDENET/`      | Core BDE prediction model                                 |
| 🧪 **BDGen**        | `bde_frag_gt/` | BDE-guided molecular optimization module                  |
| 🛠️ **Tool Models** | `tools_model/` | BDE-aware downstream models, including aweSOM and G2Retro |

---

## 🔥 Key Features

* **Bond-level BDE prediction**
* **Neural molecular representation learning**
* **BDE-guided molecular optimization**
* **BDE-aware site-of-metabolism modeling**
* **BDE-aware retrosynthesis-related modeling**
* **Multi-objective Pareto selection for molecular candidates**

---

## 📂 Repository Structure

```text
.
├── BDENET/
│   ├── bde_predictor.py
│   ├── eval_all.py
│   ├── checkpoints/
│   ├── config/
│   ├── data/
│   ├── model/
│   └── utils/
│
├── bde_frag_gt/
│   ├── __init__.py
│   ├── evaluate_properties.py
│   ├── optimizer.py
│   ├── reaction_rules.py
│   └── refine.py
│
├── frag_gt/
│   ├── frag_gt.py
│   ├── fragstore_scripts/
│   ├── src/
│   ├── data/
│   └── tests/
│
├── tools_model/
│   ├── aweSOM/
│   └── G2Retro/
│
├── data/
│   ├── fetch_guacamol.sh
│   ├── guacamol_v1_all.smiles
│   └── molecule_optimization_wetlab_pairs_20rows.csv
│
├── tests/
│   └── test_bdgen_properties.py
│
├── outputs/
├── bdenet.yml
├── run_bde_frag_gt.py
├── run_wetlab_benchmark.py
├── run_benchmark.sh
└── README.md
```

---

# 🚀 Quick Start

## 1. Create the environment

Create the conda environment from `bdenet.yml`:

```bash
conda env create -f bdenet.yml
```

Activate the environment:

```bash
conda activate bdenet
```

---


The BDENET checkpoint is placed under:

```text
BDENET/checkpoints/BDE_ckpt_new/bde_checkpoint
```

---

# 🧠 BDENET: BDE Prediction Model

The core BDENET model is located in:

```text
BDENET/
```

The main prediction interface is:

```text
BDENET/bde_predictor.py
```

BDENET predicts bond-level BDE values for molecular bonds. The predicted values can be used to characterize bond stability, identify weak bonds, and provide BDE-aware molecular features for downstream tasks.

---

## Minimal Python Example

```python
from rdkit import Chem
from BDENET.bde_predictor import BDEPredictor

checkpoint_path = "BDENET/checkpoints/DecNet/BDE_ckpt_new/bde_checkpoint"
device = "cuda:0"

predictor = BDEPredictor(
    checkpoint_path=checkpoint_path,
    device=device,
)

mol = Chem.MolFromSmiles("CCOc1ccc(CCSc2ccccc2)cc1")
bde_values = predictor.predict(mol)

print(bde_values)
```

---

# 🧪 BDGen: BDE-Guided Molecular Optimization

BDGen is implemented in:

```text
bde_frag_gt/
```

The main entry script is:

```text
run_bde_frag_gt.py
```

BDGen uses BDENET-predicted BDE values together with molecular property constraints to perform BDE-aware molecular optimization.

The default optimization supports:

* `global` optimization
* `local` optimization
* BDE-guided scoring
* property preservation
* structural filtering
* Pareto-based candidate selection

---

## Run the Benchmark

The recommended way to run the wet-lab-style optimization benchmark is:

```bash
bash run_benchmark.sh
```

The benchmark uses:

```text
data/molecule_optimization_wetlab_pairs_20rows.csv
```

as the default molecular optimization pair file.

Generated results are saved under:

```text
outputs/
```
---

## Single-Run Local Optimization

```bash
python -u run_bde_frag_gt.py \
    --mode local \
    --smiles_file data/guacamol_v1_all.smiles \
    --seed_smiles 'CCOc1ccc(CCSc2ccccc2)cc1' \
    --fragstore_path frag_gt/data/fragment_libraries/chembl_33_chemreps_std_fragstore_brics_filter2.pkl \
    --output_dir outputs/example_local \
    --population_size 200 \
    --n_mutations 200 \
    --generations 30 \
    --number_molecules 100 \
    --n_jobs 1 \
    --seed 42 \
    --maximize_bde \
    --keep_properties \
    --min_similarity 0.55 \
    --structural_filter_profile druglike_local \
    --sa_pass_threshold 6.0 \
    --logp_min 0.0 \
    --logp_max 4.0 \
    --mw_min 150.0 \
    --mw_max 550.0 \
    --tpsa_min 0.0 \
    --tpsa_max 140.0 \
    --sa_min 1.0 \
    --sa_max 6.0 \
    --selection_strategy pareto \
    --bde_checkpoint BDENET/checkpoints/DecNet/BDE_ckpt_new/bde_checkpoint \
    --device cuda:0 \
    --admet_backend none
```

---

## Single-Run Global Optimization

```bash
python -u run_bde_frag_gt.py \
    --mode global \
    --smiles_file data/guacamol_v1_all.smiles \
    --fragstore_path frag_gt/data/fragment_libraries/chembl_33_chemreps_std_fragstore_brics_filter2.pkl \
    --output_dir outputs/example_global \
    --population_size 200 \
    --n_mutations 200 \
    --generations 30 \
    --number_molecules 100 \
    --n_jobs 1 \
    --seed 42 \
    --maximize_bde \
    --keep_properties \
    --min_similarity 0.55 \
    --structural_filter_profile druglike_local \
    --sa_pass_threshold 6.0 \
    --logp_min 0.0 \
    --logp_max 4.0 \
    --mw_min 150.0 \
    --mw_max 550.0 \
    --tpsa_min 0.0 \
    --tpsa_max 140.0 \
    --sa_min 1.0 \
    --sa_max 6.0 \
    --selection_strategy pareto \
    --bde_checkpoint BDENET/checkpoints/DecNet/BDE_ckpt_new/bde_checkpoint \
    --device cuda:0 \
    --admet_backend none
```

---

# ⚙️ Main BDGen Arguments

| Argument               | Description                                 |
| ---------------------- | ------------------------------------------- |
| `--mode`               | Optimization mode: `global` or `local`      |
| `--smiles_file`        | Input SMILES file                           |
| `--seed_smiles`        | Seed molecule for local optimization        |
| `--fragstore_path`     | FragGT fragment store path                  |
| `--output_dir`         | Output directory                            |
| `--population_size`    | Population size for optimization            |
| `--n_mutations`        | Number of mutations                         |
| `--generations`        | Number of generations                       |
| `--number_molecules`   | Number of final selected molecules          |
| `--n_jobs`             | Number of parallel jobs                     |
| `--maximize_bde`       | Enable BDE maximization                     |
| `--keep_properties`    | Preserve seed-related molecular properties  |
| `--min_similarity`     | Minimum similarity constraint               |
| `--selection_strategy` | Candidate selection strategy, e.g. `pareto` |
| `--bde_checkpoint`     | BDENET checkpoint path                      |
| `--device`             | Computation device                          |
| `--admet_backend`      | ADMET backend, default can be `none`        |

---

# 🎯 Optimization Objectives

BDGen performs multi-objective candidate selection.

The default Pareto selection considers:

| Objective         | Description                               |
| ----------------- | ----------------------------------------- |
| Minimum BDE       | Avoids highly vulnerable bonds            |
| Mean BDE          | Measures overall bond energetic stability |
| QED               | Drug-likeness                             |
| LogP              | Lipophilicity range                       |
| Molecular weight  | Molecular size range                      |
| TPSA              | Polar surface area range                  |
| SA score          | Synthetic accessibility                   |
| Structural filter | Drug-like structural constraints          |
| Structural alerts | Penalizes undesirable structural alerts   |

Default property ranges:

```bash
--logp_min 0.0
--logp_max 4.0
--mw_min 150.0
--mw_max 550.0
--tpsa_min 0.0
--tpsa_max 140.0
--sa_min 1.0
--sa_max 6.0
```

---

# 📦 Outputs

Each optimization run writes results to the specified output directory.

Typical output files include:

```text
optimized_smiles.json
summary.json
summary_all_candidates.json
pareto_fronts.json
run_params.json
property_evaluation.json
seed_weakest_bond.json
seed_bde_hot_bonds.json
```

| File                          | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `optimized_smiles.json`       | Final selected optimized molecules                  |
| `summary.json`                | Summary of final selected molecules                 |
| `summary_all_candidates.json` | Summary of all generated candidates                 |
| `pareto_fronts.json`          | Pareto front information                            |
| `run_params.json`             | Parameters used in the run                          |
| `property_evaluation.json`    | Molecular property evaluation results               |
| `seed_weakest_bond.json`      | Weakest bond information of the seed molecule       |
| `seed_bde_hot_bonds.json`     | BDE-sensitive bond information of the seed molecule |

---

# 🛠️ BDE-Aware Tool Models

The downstream tool models are located in:

```text
tools_model/
```

Current modules:

```text
tools_model/
├── aweSOM/
└── G2Retro/
```

## 🔥 BDE-aware aweSOM

The BDE-aware aweSOM module is used for site-of-metabolism-related modeling with additional bond-level BDE information.

## 🔄 BDE-aware G2Retro

The BDE-aware G2Retro module is used for retrosynthesis-related modeling with bond-level energetic information.

---


# ✅ Tests

Run tests from the repository root:

```bash
pytest tests/
```

A basic BDGen property test is provided in:

```text
tests/test_bdgen_properties.py
```

---


# 📄 Citation

If you use BDENET in your research, please cite:

---

# 📜 License

This project is released for academic research use.
