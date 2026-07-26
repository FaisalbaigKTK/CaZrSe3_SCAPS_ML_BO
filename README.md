# CaZrSe₃/CdZnS Solar Cell — SCAPS-ML + Bayesian Optimisation Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Official code and dataset repository for:**

> Khattak, Y.H., Baig, F., Marí Soucase, B. *"Active Learning and Explainable Machine Intelligence
> for Lead-Free Chalcogenide Perovskite Photovoltaics: Gaussian Process Bayesian Optimisation and
> SHAP Interpretability Applied to CaZrSe₃/CdZnS Solar Cells."*
> *[Journal Name]*, 2025. DOI: [paper DOI]

---

## What this repository contains

This repository provides the **complete, reproducible pipeline** from raw SCAPS-1D simulation
output to publication-quality figures:

```
SCAPS-1D .iv files  →  Parse  →  Feature Engineering  →  Bayesian Optimisation
                                                      ↓
                                               ML Training (RF / XGBoost / ANN)
                                                      ↓
                                            SHAP Analysis  →  Paper Figures
```

### Key results reproduced by this code

| Result | Value |
|--------|-------|
| Dataset size | 12,100 SCAPS-1D simulations |
| Optimal PCE | **10.4927%** |
| Optimal Voc | 0.5388 V |
| Optimal Jsc | 27.43 mA cm⁻² |
| Optimal FF | 70.99% |
| BO evaluations used | **604 (95.0% saving)** |
| Best surrogate R² | >0.997 (XGBoost, all targets) |
| Top SHAP feature | log₁₀(Nₐ) — 53.6% of total importance |

---

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/YousafKhattak/CaZrSe3-SCAPS-ML-BO.git
cd CaZrSe3-SCAPS-ML-BO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline on the provided dataset
```bash
python run_all.py
```

### 4. Run only Bayesian optimisation (after step 2)
```bash
python src/step2b_bayesian_optimizer.py
```

### 5. Run individual steps
```bash
python src/step1_parse_iv.py         # Parse SCAPS .iv files
python src/step2_prepare_features.py # Feature engineering
python src/step3_train_models.py     # Train RF, XGBoost, ANN
python src/step4_shap_analysis.py    # SHAP explainability
python src/step5_report.py           # Generate paper numbers + figures
```

---

## Repository structure

```
CaZrSe3-SCAPS-ML-BO/
│
├── README.md                        # This file
├── LICENSE                          # MIT licence
├── requirements.txt                 # Python dependencies
├── run_all.py                       # Master pipeline runner
├── environment.yml                  # Conda environment (alternative)
│
├── configs/
│   └── config.yaml                  # All adjustable parameters in one place
│
├── src/                             # Source code — all pipeline steps
│   ├── __init__.py
│   ├── step1_parse_iv.py            # Step 1: Parse SCAPS .iv files
│   ├── step2_prepare_features.py    # Step 2: Feature engineering + log-transform
│   ├── step2b_bayesian_optimizer.py # Step 2b: GP-BO active learning (Matérn 5/2 + EI)
│   ├── step3_train_models.py        # Step 3: RF, XGBoost, ANN training + CV
│   ├── step4_shap_analysis.py       # Step 4: SHAP global + 2nd-order interactions
│   └── step5_report.py              # Step 5: Paper numbers + combined figures
│
├── data/
│   ├── raw/                         # Original SCAPS-1D .iv output files
│   │   └── CaZrSe3_sweep_12100.iv  # Full 12,100-point parametric sweep
│   ├── processed/                   # Cleaned datasets (auto-generated)
│   │   ├── clean_data.csv           # After Step 2 (12,100 rows × 8 columns)
│   │   └── feature_metadata.csv     # Feature names and display labels
│   └── results/                     # All numerical outputs (auto-generated)
│       ├── bo_optimal_params.json   # BO-found optimum device parameters
│       ├── bo_convergence.csv       # Best PCE vs evaluation number
│       ├── SHAP_importance_table.csv# SHAP global feature importance (Table 8)
│       └── paper_numbers.json       # All numbers needed for the paper
│
├── figures/
│   ├── main/                        # Main paper figures (Figs 1–13)
│   └── supplementary/               # Supplementary figures (S1–S3)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset statistics and distributions
│   ├── 02_bo_analysis.ipynb         # Bayesian optimisation analysis
│   └── 03_shap_deep_dive.ipynb      # Extended SHAP analysis
│
├── tests/
│   ├── test_parser.py               # Unit tests for .iv file parser
│   ├── test_bo.py                   # Unit tests for GP-BO components
│   └── test_models.py               # Unit tests for ML models
│
└── docs/
    ├── SCAPS_setup.md               # How to set up SCAPS-1D for new sweeps
    ├── parameter_guide.md           # Physical parameter ranges and rationale
    └── adding_new_materials.md      # How to adapt this for BaZrSe₃, CBTS, etc.
```

---

## Device specification

The simulated device is **FTO / CdZnS / CaZrSe₃ / Back contact** under AM 1.5G, 300 K, 1000 W m⁻².

| Layer | Eg (eV) | χ (eV) | Role |
|-------|---------|--------|------|
| FTO | 3.50 | 4.00 | Front contact |
| CdZnS | 2.80 | 4.20 | ETL (buffer) |
| CaZrSe₃ | 1.35 | 3.80 | Absorber (p-type) |
| Back contact | — | — | φ = 5.10 eV, ohmic |

### Swept parameters (12,100 = 11 × 11 × 10 × 10)

| Parameter | Range | Grid points |
|-----------|-------|-------------|
| CaZrSe₃ thickness | 0.10–1.50 µm | 11 (linear) |
| CaZrSe₃ Nₐ | 10¹³–10¹⁸ cm⁻³ | 11 (log) |
| CdZnS thickness | 0.02–0.20 µm | 10 (linear) |
| CdZnS Nᴅ | 10¹⁵–10¹⁸ cm⁻³ | 10 (log) |

---

## Citing this work

If you use this code or dataset, please cite:

```bibtex
@article{khattak2025cazrse3,
  title   = {Active Learning and Explainable Machine Intelligence for Lead-Free
             Chalcogenide Perovskite Photovoltaics: Gaussian Process Bayesian
             Optimisation and SHAP Interpretability Applied to CaZrSe3/CdZnS Solar Cells},
  author  = {Khattak, Yousaf Hameed and Baig, Faisal and Mar{\'i} Soucase, Bernab{\'e}},
  journal = {[Journal Name]},
  year    = {2025},
  doi     = {[paper DOI]}
}
```

And the dataset:

```bibtex
@dataset{khattak2025cazrse3_dataset,
  title     = {CaZrSe3/CdZnS SCAPS-1D Parametric Dataset (12,100 simulations)},
  author    = {Khattak, Yousaf Hameed and Baig, Faisal and Mar{\'i} Soucase, Bernab{\'e}},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## Acknowledgements

SCAPS-1D was generously provided by Prof. Marc Burgelman, University of Ghent, Belgium.
Please cite: Burgelman M. et al., *Thin Solid Films* **2000**, 361, 527–532.
https://doi.org/10.1016/S0040-6090(99)00825-1

---

## Licence

MIT — see [LICENSE](LICENSE). The dataset is released under CC BY 4.0.

---

## Contact

**Dr. Yousaf Hameed Khattak** (Corresponding Author)  
Centre for Advanced Electronics and Photovoltaic Engineering (CAEPE)  
International Islamic University Islamabad, Pakistan  
📧 yousaf.hameedk@gmail.com
