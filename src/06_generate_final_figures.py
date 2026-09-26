# Final publication visualization suite for the Gamow alpha decay project
# Plots Geiger-Nuttall trends, global predictive linearity, and residual variance

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "output" / "final_gamow_results.csv"
OUTPUT = PROJECT / "output"

print("Loading final results...")
df = pd.read_csv(INPUT)
print(f"Nuclei: {len(df)}")

# Figure 1: Geiger-Nuttall law trend tracking (Q_alpha vs Half-Life)
plt.figure(figsize=(7, 5))
plt.scatter(df["Qalpha_keV"], df["log10_half_life"], alpha=0.7, edgecolors="none")

plt.xlabel(r"Alpha decay energy $Q_{\alpha}$ (keV)")
plt.ylabel(r"$\log_{10}(T_{1/2} / \mathrm{s})$")
plt.title("Experimental Alpha Decay Trend")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

fig1_path = OUTPUT / "figure1_Qalpha_vs_halflife.png"
plt.savefig(fig1_path, dpi=300)
plt.close()

# Figure 2: Linear prediction tracking vs Experimental data points
x = df["log10_half_life"]
y = df["log10_Gamow_corrected"]

plt.figure(figsize=(7, 7))
plt.scatter(x, y, alpha=0.7, edgecolors="none")

# Dynamic axis limits calculation for identity reference line
minimum = min(x.min(), y.min())
maximum = max(x.max(), y.max())
plt.plot([minimum, maximum], [minimum, maximum], color="red", linestyle="--")

plt.xlabel(r"Experimental $\log_{10}(T_{1/2} / \mathrm{s})$")
plt.ylabel(r"Gamow predicted $\log_{10}(T_{1/2} / \mathrm{s})$")
plt.title("Gamow Model Prediction vs Experimental Data")
plt.grid(True, linestyle=":", alpha=0.6)
plt.axis("equal")
plt.tight_layout()

fig2_path = OUTPUT / "figure2_prediction_vs_experiment.png"
plt.savefig(fig2_path, dpi=300)
plt.close()

# Figure 3: Residual analysis tracking variance across Mass Number A
residual = df["log10_Gamow_corrected"] - df["log10_half_life"]

plt.figure(figsize=(7, 5))
plt.scatter(df["A"], residual, alpha=0.7, edgecolors="none")
plt.axhline(0, color="black", linestyle="-", alpha=0.5)

plt.xlabel("Mass number A")
plt.ylabel("Residual (Predicted - Experimental)")
plt.title("Gamow Model Residual Distribution")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

fig3_path = OUTPUT / "figure3_residual_analysis.png"
plt.savefig(fig3_path, dpi=300)
plt.close()

print("\nFinal publication figures generated successfully:")
print(f"-> {fig1_path}")
print(f"-> {fig2_path}")
print(f"-> {fig3_path}\n")
