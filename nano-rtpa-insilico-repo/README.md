# Nano-rtPA In-Silico Platform for Stroke Thrombolysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://doi.org/)

## Overview

Comprehensive in-silico modeling platform for evaluating carboxymethyl dextran (CMD) nanoparticle-encapsulated rtPA (Nano-rtPA) versus conventional free rtPA in acute ischemic stroke. Integrates mechanistic ordinary differential equation (ODE) modeling, external validation with real-world datasets, robust statistical analysis, and cutting-edge machine learning methods.

**Key Features:**
- ✅ Calibrated ODE model with external validation (NIHSS_802, IST datasets)
- ✅ Virtual randomized controlled trial (N=1000, powered to 90%)
- ✅ Comprehensive robustness analysis (OWSA, PSA, subgroups)
- ✅ Clinical realism modeling (safety/sICH, dose-response, nanocarrier comparison)
- ✅ Adaptive group sequential trial design
- ✅ **Machine Learning Enhancement:**
  - Bayesian Neural Networks for uncertainty quantification
  - Gaussian Process Regression for dose optimization
  - Causal Forest for heterogeneous treatment effects

## Project Structure

```
nano-rtpa-insilico-repo/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── 01_ode_model.py               # Core ODE model implementation
│   ├── 02_external_validation.py     # NIHSS_802 & IST validation
│   ├── 03_virtual_trial.py           # Powered RCT simulation
│   ├── 04_tier1_robustness.py        # OWSA, PSA, subgroups
│   ├── 05_tier2_clinical.py          # Safety, dose-response, nanocarriers
│   ├── 06_trial_design.py            # Adaptive GSD optimization
│   ├── 07_ml_bayesian_nn.py          # Bayesian Neural Network
│   ├── 08_ml_gaussian_process.py     # GP dose-response
│   ├── 09_ml_causal_forest.py        # Heterogeneous treatment effects
│   └── utils.py                       # Helper functions
│
├── data/                              # Datasets
│   ├── raw/                           # Original data (user-uploaded)
│   │   ├── NIHSS_MCA_side_bias.csv   
│   │   ├── 13063_2010_637_MOESM1_ESM.CSV
│   │   └── prego_table_s3_per_study.csv
│   └── processed/                     # Generated results
│       ├── external_validation.npz
│       ├── tier1_final.npz
│       ├── tier2_results.npz
│       ├── trial_design_final.npz
│       ├── ml_bnn_results.npz
│       ├── ml_gp_results.npz
│       └── ml_cf_results.npz
│
├── figures/                           # Publication-quality figures (350 DPI)
│   ├── nanoparticle_schematic_v2.png
│   ├── mechanistic_pathway_clean.png
│   ├── pkpd_model_final.png
│   ├── external_validation.png
│   ├── tier1_robustness_final.png
│   ├── tier2_clinical_realism.png
│   ├── trial_design_final.png
│   ├── ml_bayesian_neural_network.png
│   ├── ml_gaussian_process.png
│   └── ml_causal_forest.png
│
├── docs/                              # Additional documentation
│   ├── METHODS.md                     # Detailed methodology
│   ├── RESULTS_SUMMARY.md             # Key findings
│   └── CITATION.bib                   # Citation information
│
└── notebooks/                         # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb

```

## Key Results

### Efficacy (Primary Outcome: mRS 0-2 at 90 days)
| Treatment | Response Rate | Relative Risk | 95% CI | p-value |
|-----------|---------------|---------------|--------|---------|
| Free rtPA | 20.8% | 1.00 (ref) | — | — |
| Nano-rtPA | 26.6% | **1.28** | [1.01, 1.77] | 0.037 |

**Bayesian Analysis:** P(Nano > Free) = 98.5%

### Safety (Symptomatic Intracranial Hemorrhage)
- Free rtPA: 6.5% → Nano-rtPA: **4.5%** (30% reduction)
- **Net Clinical Benefit:** +5.9% (efficacy + safety combined)

### Dose Optimization (Gaussian Process Regression)
- **59% dose reduction** possible: Nano 0.37 mg/kg ≈ Free 0.90 mg/kg
- Optimal nano dose: 1.37 mg/kg (24% efficacy)

### Personalized Medicine (Causal Forest)
- Mean CATE: 9.5% absolute benefit
- **High-benefit subgroup** (top 25%): 34.9% benefit
  - Profile: Age <60, NIHSS 4-7, OTT <120 min

### Trial Design
- **Recommended:** Adaptive Group Sequential Design
- Sample size: N=800 (400/arm) for 80% power
- Expected savings: 36% under H1, 49% under H0

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nano-rtpa-insilico-repo.git
cd nano-rtpa-insilico-repo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Complete Analysis Pipeline

```bash
# Execute all analyses in sequence
python src/01_ode_model.py
python src/02_external_validation.py
python src/03_virtual_trial.py
python src/04_tier1_robustness.py
python src/05_tier2_clinical.py
python src/06_trial_design.py
python src/07_ml_bayesian_nn.py
python src/08_ml_gaussian_process.py
python src/09_ml_causal_forest.py
```

### 2. Generate Individual Figures

```python
# Example: Generate nanoparticle schematic
from src.utils import generate_nanoparticle_schematic
generate_nanoparticle_schematic(output_path='figures/nanoparticle.png', dpi=350)
```

### 3. Load Existing Results

```python
import numpy as np

# Load Tier 1 robustness results
tier1_data = np.load('data/processed/tier1_final.npz')
psa_rr = tier1_data['psa_rr']
print(f"PSA mean RR: {psa_rr.mean():.2f}")
print(f"95% CI: [{np.percentile(psa_rr, 2.5):.2f}, {np.percentile(psa_rr, 97.5):.2f}]")
```

## Dependencies

**Core:**
- Python ≥3.8
- NumPy ≥1.21.0
- SciPy ≥1.7.0
- Pandas ≥1.3.0
- Matplotlib ≥3.4.0

**Machine Learning:**
- scikit-learn ≥1.0.0

See `requirements.txt` for complete list with pinned versions.

## Methodology

### ODE Model (Host Stroke Framework)
- Clot lysis dynamics: dV/dt = -k_lysis × V × [rtPA]
- Functional outcome mapping: NIHSS → infarct volume → mRS
- Calibrated to meta-analysis data (Prego et al., k=18 studies)

### External Validation
- **NIHSS_802 dataset** (Mah et al., 2021): NIHSS ↔ Volume correlation validated (r=0.58, error=0.41%)
- **IST dataset** (N=19,435): Mortality baselines, outcome distributions

### Machine Learning Framework

**Bayesian Neural Network (BNN):**
- Architecture: Ensemble (N=100) with MC dropout
- Purpose: Epistemic + aleatoric uncertainty quantification
- Calibration: 95% CI coverage ≈ 100%

**Gaussian Process Regression (GPR):**
- Kernel: Radial Basis Function (RBF)
- Application: Non-parametric dose-response optimization
- Key finding: 59% dose reduction possible

**Causal Forest:**
- Method: T-learner with Random Forest (N=200 trees)
- Purpose: Conditional Average Treatment Effect (CATE) estimation
- Identifies 3.5× benefit in optimal subgroup

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{nano_rtpa_insilico_2025,
  title={Machine Learning-Enhanced In-Silico Evaluation of Nano-rtPA for Acute Ischemic Stroke: A Comprehensive Modeling Study},
  author={[Your Name] and Collaborators},
  journal={[Journal Name]},
  year={2025},
  doi={pending}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## Contact

**Principal Investigator:** [Your Name]  
**Institution:** [Your Institution]  
**Email:** [your.email@institution.edu]

## Acknowledgments

- NIHSS_802 dataset: Mah Y-H, et al. (2021)
- IST-3 trial data: Sandercock et al. (2012)
- Prego meta-analysis: Prego C, et al. (2010)
- Funding: [Your Funding Source]

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Status:** Publication-ready
