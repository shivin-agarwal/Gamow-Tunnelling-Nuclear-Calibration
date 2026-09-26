"""
01_build_dataset.py

Gamow Alpha Decay Research Project

Parses and matches AME2020 atomic mass evaluations with NUBASE2020 decay properties
to isolate a clean experimental dataset for Gamow quantum tunnelling calculations.

Input data tracks:
    - data/rct1.mas20.txt (AME2020 reaction energies)
    - data/nubase_4.mas20.txt (NUBASE2020 ground state properties)

Output:
    - output/alpha_decay_dataset.csv
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# Paths and project structure
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
OUTPUT.mkdir(exist_ok=True)

RCT_FILE = DATA / "rct1.mas20.txt"
NUBASE_FILE = DATA / "nubase_4.mas20.txt"

# Time conversion constants (seconds per unit) for half-life normalization
TIME_CONVERSION = {
    "ys": 1e-24,
    "zs": 1e-21,
    "as": 1e-18,
    "fs": 1e-15,
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": 365.25 * 24.0 * 3600.0
}


def clean_number(x):
    """Cleans raw string values from AME/NUBASE tables, dropping non-numeric markers."""
    x = x.strip()
    if x in ["", "*"]:
        return np.nan
    
    # '#' denotes estimated values derived from systematic trends rather than direct experiment
    x = x.replace("#", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def even_even(A, Z):
    """Returns True if the nucleus has both even Z and even N."""
    N = A - Z
    return (A % 2 == 0) and (Z % 2 == 0)


def read_rct_file(filename):
    """Parses fixed-width AME2020 format to extract Q-alpha values and uncertainties."""
    print("Parsing AME2020 data...")
    records = []

    with open(filename, "r", encoding="latin1") as f:
        lines = f.readlines()

    # Locate table header marker dynamically
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("1 A  elt")), None)
    if start is None:
        raise ValueError("Malformed AME file: Table header marker not found")

    for line in lines[start:]:
        if not line.strip():
            continue

        try:
            # Fixed-width slicing based on official AME2020 layout standards
            A = int(line[1:4])
            element = line[5:8].strip()
            Z = int(line[8:11])
            q_alpha = clean_number(line[56:68])
            q_unc = clean_number(line[68:78])

            if not np.isnan(q_alpha):
                records.append({
                    "A": A,
                    "Z": Z,
                    "Element": element,
                    "Qalpha_keV": q_alpha,
                    "Qalpha_unc_keV": q_unc
                })
        except (ValueError, IndexError):
            continue  # Silently skip header trailing lines or corrupted rows

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["A", "Z"])
    print(f"-> AME nuclei loaded: {len(df)}")
    return df


def read_nubase_file(filename):
    """Parses NUBASE2020 format to extract half-lives and decay mode string markers."""
    print("Parsing NUBASE2020 ground states...")
    records = []

    with open(filename, "r", encoding="latin1") as f:
        for line in f:
            if line.startswith("#") or len(line) < 120:
                continue

            try:
                A = int(line[0:3])
                Z = int(line[4:7])
            except ValueError:
                continue

            # Skip isomers to target strictly ground states for base decay constraints
            if line[16].strip() != "":
                continue

            half = line[69:78].strip()
            unit = line[78:80].strip()
            decay = line[119:].strip()

            records.append({
                "A": A,
                "Z": Z,
                "HalfLife_raw": half,
                "Unit": unit,
                "Decay": decay
            })

    df = pd.DataFrame(records)
    print(f"-> NUBASE ground states loaded: {len(df)}")
    return df


def half_life_seconds(value, unit):
    """Converts multi-unit raw half-lives into scientific standard SI seconds."""
    if value in ["stbl", "p-unst", ""]:
        return np.nan

    value = value.replace("#", "")
    try:
        value = float(value)
    except ValueError:
        return np.nan

    factor = TIME_CONVERSION.get(unit)
    if factor is None:
        return np.nan

    return value * factor


def is_alpha_dominant(decay):
    """Isolates isotopes where alpha decay represents the primary branching pathway (>= 95%)."""
    if pd.isna(decay):
        return False

    decay = decay.strip()
    
    # Target regex pattern for alpha branching percentages (e.g., A=100 or A~98.5)
    match = re.search(r"A[=~](\d+\.?\d*)", decay)
    if match:
        alpha_branch = float(match.group(1))
        if alpha_branch >= 95.0:
            return True

    return False


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Ingest raw database formats
    ame_df = read_rct_file(RCT_FILE)
    nubase_df = read_nubase_file(NUBASE_FILE)

    # 2. Process temporal baselines
    nubase_df["HalfLife_seconds"] = [
        half_life_seconds(val, unit) 
        for val, unit in zip(nubase_df["HalfLife_raw"], nubase_df["Unit"])
    ]

    # 3. Filter for clean alpha decay tracks (ignoring beta/fission dominant channels)
    alpha_dominant_df = nubase_df[nubase_df["Decay"].apply(is_alpha_dominant)]
    print(f"Alpha-dominant emitters found: {len(alpha_dominant_df)}")

    # 4. Coalesce energy values with temporal tracking metrics
    merged_df = pd.merge(ame_df, alpha_dominant_df, on=["A", "Z"], how="inner")

    # 5. Restrict to even-even systems to negate complex ground-state hindrance considerations
    merged_df["even_even"] = [even_even(a, z) for a, z in zip(merged_df["A"], merged_df["Z"])]
    filtered_df = merged_df[merged_df["even_even"]].copy()

    # 6. Apply physical cutoffs (Z >= 50 avoids lighter structural anomalies)
    filtered_df = filtered_df[filtered_df["Z"] >= 50]

    # 7. Drop incomplete records where critical variables are missing
    filtered_df = filtered_df.dropna(subset=["Qalpha_keV", "HalfLife_seconds"])

    # 8. Compute physical scaling logarithm for Geiger-Nuttall analysis
    filtered_df["log10_half_life"] = np.log10(filtered_df["HalfLife_seconds"])

    # Save formatted research outputs
    output_path = OUTPUT / "alpha_decay_dataset.csv"
    filtered_df.to_csv(output_path, index=False)

    print("\n" + "="*40)
    print("SUCCESS: VALIDATED COULOMB BARRIER DATASET BUILT")
    print(f"Destination: {output_path}")
    print(f"Final valid nuclei records count: {len(filtered_df)}")
    print("="*40 + "\n")
    
    print(filtered_df.head())
