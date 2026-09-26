# Gamow-Tunnelling-Nuclear-Calibration
Global Calibration of a 1D Gamow Quantum Tunnelling Model for Heavy Even-Even Alpha-Emitters using AME2020 and NUBASE2020.
# Data Track Baselines
The data execution pipelines assume raw fixed-width source tracks are stored within the `data/` directory:
1. *AME2020 Matrix: data/rct1.mas20.txt` (Atomic Mass Evaluation reaction energies)
2. **NUBASE2020 Inventory:** `data/nubase_4.mas20.txt` (Nuclear ground-state structural configurations)

# Analytical Processing Sequence

To reproduce the publication figures and model datasets, execute the pipeline components sequentially:

```bash
# Step 1: Parse databases, apply physics constraints, and clean baseline data
python src/01_build_dataset.py

# Step 2: Run numerical integration for pure WKB tunneling model predictions
python src/02_gamow_model.py

# Step 3: Optimize global intercept limits to fit empirical preformation constants
python src/03_fit_preformation.py

# Step 4: Output publication graphics suite and residual analyses
python src/04_generate_final_figures.py
```

# Dataset Filtering Criteria
To enforce physical consistency and isolate structural mechanics, `01_build_dataset.py` strips experimental metrics against explicit constraints:
* Decay Track Dominance: Restricts evaluations to isotopes where alpha emission forms \(\ge 95\%\) of the total branching fraction.
* Ground State Transitions: Filters strictly for even-even configurations (\(Z \pmod 2 = 0, N \pmod 2 = 0\)) to bypass complex rotational/vibrational ground-state hindrance contributions.
* Heavy Nuclei Bounds: Restricts evaluations to \(Z \ge 50\) to eliminate lighter structural anomalies and guarantee Coulomb barrier validity limits.

# Production Artifact Targets
Successful compilation yields data files and graphics inside the `output/` ecosystem:
* `output/alpha_decay_dataset.csv` - Consolidated raw physics parameters.
* `output/gamow_results.csv` - Semiclassical lifetime model logs.
* `output/final_gamow_results.csv` - Model logs optimized against fitted clustering metrics.
* `output/figure1_Qalpha_vs_halflife.png` - Geiger-Nuttall law trend plots.
* `output/figure2_prediction_vs_experiment.png` - Scatter metrics monitoring model prediction parity.
* `output/figure3_residual_analysis.png` - Residual variance distribution tracking across mass numbers (\(A\)).

# Technical Dependencies
Calculations require a standard Python 3 runtime environment configured with the following dependencies:
* `numpy` (Numerical array manipulation tracking)
* `pandas` (Tabular database structure handling)
* `scipy` (Quad pack numerical adaptive Gauss-Kronrod integrations)
* `matplotlib` (Vector visualization engines)
