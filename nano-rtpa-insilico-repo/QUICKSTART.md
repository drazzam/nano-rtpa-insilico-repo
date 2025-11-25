# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nano-rtpa-insilico-repo.git
cd nano-rtpa-insilico-repo

# Install dependencies
pip install -r requirements.txt
```

## Run Complete Analysis

```bash
# Execute the full pipeline
python src/complete_analysis_pipeline.py
```

Expected output:
```
======================================================================
NANO-rtPA IN-SILICO PLATFORM: COMPLETE ANALYSIS
======================================================================

[1/6] Initializing ODE model...
✓ Model initialized with calibrated parameters

[2/6] Performing external validation...
✓ External validation: r=0.578, error=0.41%

[3/6] Simulating virtual randomized controlled trial...
✓ Virtual RCT complete:
  Free rtPA: 20.8%
  Nano-rtPA: 26.6%
  RR: 1.28

[4/6] Conducting probabilistic sensitivity analysis...
✓ PSA complete: Mean RR=1.341, 95% CI=[1.01, 1.77]

[5/6] Training Bayesian Neural Network...
✓ BNN trained: Mean uncertainty = 0.479

[6/6] Optimizing dose-response with Gaussian Process...
✓ GP optimization complete: Optimal dose = 1.37 mg/kg

======================================================================
ANALYSIS COMPLETE - All results saved to data/processed/
======================================================================
```

## Load Existing Results

```python
import numpy as np

# Load PSA results
tier1_data = np.load('data/processed/tier1_final.npz')
psa_rr = tier1_data['psa_rr']

print(f"Mean RR: {psa_rr.mean():.3f}")
print(f"95% CI: [{np.percentile(psa_rr, 2.5):.2f}, {np.percentile(psa_rr, 97.5):.2f}]")
print(f"P(RR > 1.0): {(psa_rr > 1.0).mean()*100:.1f}%")
```

## View Generated Figures

All publication-quality figures (350 DPI) are in `figures/`:

- `nanoparticle_schematic_v2.png` - Nanoparticle architecture
- `mechanistic_pathway_clean.png` - Therapeutic cascade
- `pkpd_model_final.png` - PK/PD compartment model
- `external_validation.png` - Validation with NIHSS_802/IST
- `tier1_robustness_final.png` - Sensitivity analysis
- `tier2_clinical_realism.png` - Safety/dose-response
- `trial_design_final.png` - Adaptive GSD
- `ml_bayesian_neural_network.png` - BNN uncertainty
- `ml_gaussian_process.png` - GP dose optimization
- `ml_causal_forest.png` - Heterogeneous effects

## Troubleshooting

**Problem:** `FileNotFoundError` for data files

**Solution:** Ensure data files are in `data/raw/`:
```bash
ls data/raw/
# Should show:
# NIHSS_MCA_side_bias.csv
# 13063_2010_637_MOESM1_ESM.CSV
# prego_table_s3_per_study.csv
```

**Problem:** Import errors

**Solution:** Verify dependencies:
```bash
pip list | grep -E 'numpy|scipy|pandas|matplotlib|scikit'
```

## Next Steps

1. Explore `docs/METHODS.md` for detailed methodology
2. Review `docs/RESULTS_SUMMARY.md` for key findings
3. Adapt code in `src/` for your specific use case
4. Generate custom figures by modifying plotting code

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: [your.email@institution.edu]
- Cite using `docs/CITATION.bib`
