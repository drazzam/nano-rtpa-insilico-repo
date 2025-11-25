# Detailed Methodology

## 1. ODE Model Framework

### Clot Lysis Dynamics
```
dV/dt = -k_lysis × V × [rtPA]
```

Where:
- V = clot volume (mL)
- k_lysis = lysis rate constant (calibrated to 0.50 h⁻¹)
- [rtPA] = plasma rtPA concentration

### Functional Outcome Mapping
```
logit(P(mRS ≤ 2)) = α₀ - 0.08×NIHSS + β×Treatment
```

Calibrated parameters:
- α₀ = -0.50 (baseline logit)
- β = 0.393 (treatment effect for Nano-rtPA)

## 2. External Validation

### NIHSS_802 Dataset (N=802)
- **Source:** Mah Y-H, et al. Scientific Reports, 2021
- **Validation metric:** NIHSS ↔ Infarct Volume correlation
- **Result:** r = 0.58 (p < 0.001), prediction error = 0.41%

### IST Dataset (N=19,435)
- **Source:** Sandercock et al. Lancet, 2012
- **Purpose:** Mortality baselines and outcome distributions
- **Limitation:** Pre-thrombolytic era (no rtPA arm)

## 3. Machine Learning Framework

### Bayesian Neural Network
- **Architecture:** Ensemble (N=100 estimators)
- **Training:** Bootstrap sampling with MC dropout
- **Uncertainty:** Epistemic (model) + Aleatoric (data)
- **Calibration:** 95% CI coverage validated

### Gaussian Process Regression
- **Kernel:** Radial Basis Function (RBF)
- **Hyperparameters:** 
  - Length scale: 0.3
  - Signal variance: 100
  - Noise variance: 9 (Free), 6.25 (Nano)
- **Application:** Dose-response optimization

### Causal Forest
- **Method:** T-learner with Random Forest
- **Trees:** N=200, max depth=12
- **Purpose:** CATE (Conditional Average Treatment Effect)
- **Validation:** Out-of-sample predictions

## 4. Statistical Analysis

### Probabilistic Sensitivity Analysis (PSA)
- **Iterations:** N=2,000
- **Parameter distributions:**
  - α ~ N(-1.224, 0.10)
  - β ~ N(0.393, 0.06)
- **Output:** RR distribution with 95% credible interval

### Subgroup Analysis
Stratified by:
1. Age: <65 vs ≥65 years
2. Stroke severity: NIHSS <10, 10-20, >20
3. Onset-to-treatment: <90, 90-180, >180 min

## 5. Trial Design

### Group Sequential Design
- **Type:** O'Brien-Fleming boundaries
- **Looks:** 2 (interim at 50%, final at 100%)
- **Efficacy stopping:** Z > 2.80 (interim), Z > 1.96 (final)
- **Futility stopping:** Conditional power < 20%

### Sample Size Calculation
Based on Fleiss formula with continuity correction:
- 80% power: N = 1,686 (843/arm)
- 90% power: N = 2,256 (1,128/arm)

## References

1. Prego C, et al. J Control Release. 2010;147(3):408-417.
2. Mah Y-H, et al. Sci Rep. 2021;11:14641.
3. Sandercock et al. Lancet. 2012;379(9834):2352-2363.
4. Emberson J, et al. Lancet. 2014;384(9958):1929-1935.
