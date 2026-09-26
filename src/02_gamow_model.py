# Gamow Quantum Tunnelling Model for Alpha Decay
# Implements the WKB approximation to calculate alpha-decay half-lives

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "output" / "alpha_decay_dataset.csv"
OUTPUT = PROJECT / "output"

R0 = 1.2e-15
A_ALPHA = 4
Z_ALPHA = 2
M_ALPHA = 6.644657e-27
HBAR = 1.054571817e-34
K_COULOMB = 8.9875517923e9
E_CHARGE = 1.602176634e-19
KEV_TO_J = 1.602176634e-16

def gamow_log10_half_life(A, Z, Q_keV):
    if Q_keV <= 0:
        return np.nan

    A_daughter = A - 4
    Z_daughter = Z - 2
    Q = Q_keV * KEV_TO_J
    R = R0 * (A_ALPHA**(1/3) + A_daughter**(1/3))
    b = (K_COULOMB * Z_ALPHA * Z_daughter * E_CHARGE**2) / Q

    if b <= R:
        return np.nan

    def integrand(r):
        v_coulomb = (K_COULOMB * Z_ALPHA * Z_daughter * E_CHARGE**2) / r
        delta_v = v_coulomb - Q
        return np.sqrt(2 * M_ALPHA * delta_v) if delta_v > 0 else 0.0

    integral, _ = quad(integrand, R, b)
    G = integral / HBAR
    velocity = np.sqrt(2 * Q / M_ALPHA)
    frequency = velocity / (2 * R)

    log10_lambda = np.log10(frequency) - (2 * G) / np.log(10)
    return np.log10(np.log(2)) - log10_lambda

if __name__ == "__main__":
    print("Loading nuclear parameters...")
    df = pd.read_csv(INPUT)

    print("Running Gamow calculations...")
    df["log10_Gamow"] = [
        gamow_log10_half_life(row.A, row.Z, row.Qalpha_keV) 
        for row in df.itertuples(index=False)
    ]
    df["Gamow_half_life_seconds"] = 10 ** df["log10_Gamow"]

    output_file = OUTPUT / "gamow_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Calculations finalized and written to: {output_file}")
