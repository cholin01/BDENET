# 🧬 BDENET：Bond Energetics as a Physical Coordinate for Molecular Discovery and Design

**BDENET** is a bond-level deep learning framework for **bond dissociation energy (BDE) prediction** and **BDE-aware molecular modeling**.

The repository provides a core BDE prediction model, a BDE-guided molecular optimization module, and several downstream BDE-aware molecular modeling tools.



<p align="center">
  <img 
    src="data/overview.PNG" 
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
| 🧪 **BDGen**        | [`BDGen/`](BDGen/) | Six-strategy BDE-guided molecular optimization |
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
├── BDGen/
│   ├── run_bdgen.py
│   ├── README.md
│   ├── requirements.txt
│   ├── assets/
│   ├── config/
│   ├── data/
│   └── core/
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
│   ├── Extra_testset.csv
│   ├── Natural_products_testset.csv
│   ├── molecule_optimization_wetlab_pairs_15_unique.csv
│   └── overview.PNG
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

checkpoint_path = "BDENET/checkpoints/BDE_ckpt_new/bde_checkpoint"
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

BDGen is distributed in [`BDGen/`](BDGen/), with a single entry point, `BDGen/run_bdgen.py`, for six optimization strategies:

* `property-only`, `bde-only`, and `joint` compare optimization objectives.
* `random`, `static`, and `dynamic` compare editing-region selection under the same joint BDE and property objective.

The package includes the inference checkpoint, BRICS fragment library, and 15 starting structures numbered MO-001 through MO-015. Dynamic editing samples from up to three distinct low-BDE regions of the current parent. All six strategies share the heavy-atom retention constraint of 70% and soft similarity guidance of 0.55. BDE objectives are evaluated over non-ring heavy-atom single bonds.

Follow the separate Linux/Python 3.12 installation instructions in the [BDGen README](BDGen/README.md). From the repository root, after activating that environment:

```bash
cd BDGen
python run_bdgen.py all --case MO-008 --profile smoke --output smoke_results --jobs 2
```

To run all 15 cases under all six strategies:

```bash
python run_bdgen.py all --output results --jobs 2
```

See the [BDGen README](BDGen/README.md) for individual strategies, effective search settings, candidate selection, output definitions, and optional post-generation ADMET evaluation. ADMET endpoints do not participate in the six-strategy optimization workflow.

## Benchmark execution and reports

`run_wetlab_benchmark.py` runs the shared BDGen entry point and exports final candidates with literature-pair metadata. `run_benchmark.sh` provides the same interface as a shell command:

```bash
bash run_benchmark.sh --strategy all --jobs 2 --output outputs/wetlab_benchmark
```

Use `--profile smoke --case MO-008` for a short functional check, or `--plan-only` to inspect the commands. To export reports from a completed run without generating candidates:

```bash
python run_wetlab_benchmark.py --report-only --output outputs/wetlab_benchmark
```

Reports are written under `<output>/benchmark_reports/`: all selected candidates, their Top-K prefix, per-case strategy summaries with actual counts and shortfalls, and literature references. Reference records are matched by canonical isomeric initial SMILES; `case_id` follows the BDGen case list and `source_pair_id` preserves the reference CSV identifier. Candidate-file hashes are verified before export, and the order in `candidates.json` is retained. `--top-k` controls only the additional report, with a default of 10.

`run_bde_frag_gt.py` forwards six-strategy commands to `BDGen/run_bdgen.py`. Install the dependencies described in [BDGen/README.md](BDGen/README.md) for all entry points.

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


# ✅ BDGen Functional Check

Use the `smoke` command above to check candidate generation and Pareto selection after installing the BDGen dependencies. Run the routing and benchmark-export checks with:

```bash
python -m unittest discover -s tests -v
```

---

# 📄 Citation

If you use BDENET in your research, please cite:

---

# 📜 License

This project is released for academic research use.
