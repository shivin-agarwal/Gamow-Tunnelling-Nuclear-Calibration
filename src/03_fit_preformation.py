# Gamow model correction: fitting the alpha preformation factor P_alpha
# Minimizes global systemic deviation between pure WKB and experimental logs

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "output" / "gamow_results.csv"
OUTPUT = PROJECT / "output"

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

def calculate_r2(actual, predicted):
    # Evaluates true R² based on residual sum of squares vs total variance
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    return 1.0 - (ss_res / ss_tot)

if __name__ == "__main__":
    print("Loading baseline model results framework...")
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing upstream dataset asset: {INPUT}")
        
    df = pd.read_csv(INPUT)
    print(f"-> Active entries loaded for calibration: {len(df)}")

    # Pure WKB systematically underestimates lifetimes because it assumes 
    # the alpha cluster is preformed at the boundary (P_alpha = 1.0).
    # Quantify the uniform additive shift required to calibrate the baseline.
    residuals = df["log10_half_life"] - df["log10_Gamow"]
    mean_shift = residuals.mean()

    # Relationship handling inverse proportionality:
    # log10_corrected = log10_Gamow - log10(P_alpha)
    log10_P_alpha = -mean_shift
    P_alpha = 10**log10_P_alpha

    print(f"\nFitted parameters:")
    print(f"Logarithmic shift (decades): +{mean_shift:.4f}")
    print(f"log10(P_alpha):               {log10_P_alpha:.4f}")
    print(f"Effective P_alpha:            {P_alpha:.4f}")

    # Adjust model projections by scaling decay constant shift
    df["log10_Gamow_corrected"] = df["log10_Gamow"] + mean_shift
    df["Gamow_corrected_half_life_seconds"] = 10 ** df["log10_Gamow_corrected"]

    # Evaluate global predictive statistical errors
    rmse_raw = calculate_rmse(df["log10_half_life"], df["log10_Gamow"])
    rmse_corrected = calculate_rmse(df["log10_half_life"], df["log10_Gamow_corrected"])
    r2_score = calculate_r2(df["log10_half_life"], df["log10_Gamow_corrected"])

    print(f"\nModel Optimization Summary:")
    print(f"Raw Gamow RMSE:        {rmse_raw:.4f}")
    print(f"Corrected Gamow RMSE:  {rmse_corrected:.4f}")
    print(f"Error Reduction:       {((rmse_raw - rmse_corrected)/rmse_raw)*100:.1f}%")
    print(f"Determination R²:      {r2_score:.4f}")

    output_file = OUTPUT / "final_gamow_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved final calibrated dataset to: {output_file}\n")
