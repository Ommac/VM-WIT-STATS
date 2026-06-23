# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Friedman Test
# File        : hypothesis_tests/friedman_test.py
# Description : Non-parametric alternative to Repeated Measures ANOVA.
#               Tests whether repeated measurements on the same subjects
#               differ significantly across three or more time points.
#               Uses scipy.stats.friedmanchisquare().
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "friedman_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Friedman Test
{SEPARATOR}
"""

# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_header():
    print(HEADER)


def print_section(title: str):
    print(f"\n{SEPARATOR2}")
    print(f"  {title}")
    print(SEPARATOR2)


def fmt_val(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, float):
        return f"{val:,.4f}"
    return str(val)


def is_effectively_constant(series: pd.Series, tol: float = 1e-10) -> bool:
    return bool((series.max() - series.min()) < tol)


def kendalls_w(chi2: float, n_subjects: int, k_measures: int) -> float:
    """
    Compute Kendall's W (coefficient of concordance) as effect size.

        W = χ²_F / (N × (k - 1))

    where:
        χ²_F = Friedman chi-square statistic
        N    = number of subjects
        k    = number of repeated measures

    Returns NaN if denominator is zero.
    Clamped to [0.0, 1.0].
    """
    denom = n_subjects * (k_measures - 1)
    if denom <= 0:
        return float("nan")
    return max(0.0, min(1.0, chi2 / denom))


def interpret_kendalls_w(w: float) -> str:
    """
    Interpret Kendall's W using standard benchmarks:
      W < 0.10 → Negligible
      W < 0.30 → Small
      W < 0.50 → Moderate
      W ≥ 0.50 → Large
    """
    if math.isnan(w):
        return "N/A"
    if w < 0.10:
        return "Negligible"
    elif w < 0.30:
        return "Small"
    elif w < 0.50:
        return "Moderate"
    else:
        return "Large"


# ======================================================================
# SAMPLE DATA GENERATOR (for testing without repeated-measures CSV)
# ======================================================================

def generate_sample_csv(path: str = "data/sample_repeated_measures.csv",
                         n: int = 30,
                         seed: int = 42):
    """
    Generate a synthetic repeated-measures dataset for Friedman Test testing.

    Columns:
        Patient_ID : unique subject identifier
        Baseline   : baseline measurement
        Month1     : 1-month follow-up
        Month3     : 3-month follow-up
        Month6     : 6-month follow-up

    Usage (standalone):
        from hypothesis_tests.friedman_test import generate_sample_csv
        generate_sample_csv("data/sample_repeated_measures.csv")
    """
    rng = np.random.default_rng(seed)
    baseline = rng.normal(loc=7.5, scale=1.5, size=n).clip(1, 10)
    month1   = baseline - rng.uniform(0.3, 1.5, size=n) + rng.normal(0, 0.3, size=n)
    month3   = month1   - rng.uniform(0.5, 2.0, size=n) + rng.normal(0, 0.4, size=n)
    month6   = month3   - rng.uniform(0.2, 1.0, size=n) + rng.normal(0, 0.3, size=n)

    df = pd.DataFrame({
        "Patient_ID": [f"PT{str(i+1).zfill(3)}" for i in range(n)],
        "Baseline"  : baseline.round(2),
        "Month1"    : month1.clip(1, 10).round(2),
        "Month3"    : month3.clip(1, 10).round(2),
        "Month6"    : month6.clip(1, 10).round(2),
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows: {n}  |  Columns: Patient_ID, Baseline, Month1, Month3, Month6")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    print("\n  NOTE: The Friedman Test requires a WIDE-FORMAT repeated-measures dataset.")
    print("        Each row = one subject. Each repeated measurement = one column.")
    print("        Example: Patient_ID | Baseline | Month1 | Month3 | Month6")
    print("\n  TIP: To generate a sample dataset for testing, run:")
    print("       from hypothesis_tests.friedman_test import generate_sample_csv")
    print("       generate_sample_csv()")

    while True:
        path = input("\n  Enter path to CSV file: ").strip()
        if not path:
            print("  [ERROR] No path entered. Please try again.")
            continue
        if not path.lower().endswith(".csv"):
            print("  [WARNING] File does not have a .csv extension. Attempting to load anyway.")
        if not os.path.isfile(path):
            print(f"  [ERROR] File not found: '{path}'. Please check the path and try again.")
            continue
        return path


def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print("  [ERROR] The CSV file is empty. Please provide a file with data.")
        return None
    except pd.errors.ParserError as exc:
        print(f"  [ERROR] Failed to parse CSV file.\n          Details: {exc}")
        return None
    except PermissionError:
        print("  [ERROR] Permission denied. Cannot read the file.")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error while loading file.\n          Details: {exc}")
        return None

    if df.empty:
        print("  [ERROR] The CSV file loaded successfully but contains no rows.")
        return None

    print(f"\n  [OK] Dataset loaded successfully.")
    print(f"       Rows    : {df.shape[0]:,}")
    print(f"       Columns : {df.shape[1]:,}")
    return df


def display_columns(df: pd.DataFrame):
    print_section("STEP 2 | AVAILABLE COLUMNS")
    print(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15}")
    print(f"  {'-'*5} {'-'*40} {'-'*15}")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        print(f"  {idx:<5} {col:<40} {str(dtype):<15}")


# ======================================================================
# COLUMN SELECTION
# ======================================================================

def select_subject_id_column(df: pd.DataFrame) -> str | None:
    print_section("STEP 3 | SELECT SUBJECT ID COLUMN")
    print("\n  The Subject ID column uniquely identifies each subject/patient.")
    print("  Example: Patient_ID, SubjectID, ID, PatientNo, etc.")

    while True:
        col = input("\n  Enter Subject ID column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

        # Check for duplicates
        dupes = df[col].duplicated().sum()
        if dupes > 0:
            print(f"  [WARNING] Subject ID column '{col}' contains {dupes:,} duplicate value(s).")
            print("            Duplicate IDs mean multiple rows per subject.")
            ans = input("            Continue anyway? This may affect results. (yes/no): ").strip().lower()
            if ans not in ("yes", "y"):
                print("            Please select a different Subject ID column.")
                continue

        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  [WARNING] {n_missing:,} missing Subject ID(s) will be excluded.")

        print(f"  [OK] Subject ID column selected: '{col}'")
        print(f"       Unique subjects: {df[col].nunique():,}")
        return col


def select_repeated_measure_columns(df: pd.DataFrame,
                                     subject_col: str) -> list | None:
    print_section("STEP 4 | SELECT REPEATED MEASURE COLUMNS")
    print("\n  Enter the names of columns representing repeated measurements.")
    print("  You need at least 3 repeated measures for the Friedman Test.")
    print("  Example: Baseline, Month1, Month3, Month6")
    print("  Type 'done' when finished entering column names.")

    selected = []
    while True:
        col = input(f"\n  Enter repeated measure column {len(selected)+1} "
                    f"(or 'done' if finished): ").strip()

        if col.lower() == "done":
            if len(selected) < 3:
                print(f"  [ERROR] You have selected only {len(selected)} column(s).")
                print("          Friedman Test requires at least 3 repeated measures.")
                continue
            break

        if not col:
            print("  [ERROR] No column name entered. Type 'done' when finished.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue
        if col == subject_col:
            print(f"  [ERROR] Cannot use Subject ID column '{subject_col}' as a repeated measure.")
            continue
        if col in selected:
            print(f"  [WARNING] Column '{col}' already selected. Skipping.")
            continue

        # Check numeric coercibility
        coerced = pd.to_numeric(df[col], errors="coerce")
        valid = coerced.notna().sum()
        if valid == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values. Cannot use as a repeated measure.")
            continue
        non_numeric = len(coerced) - valid
        if non_numeric > 0:
            print(f"  [WARNING] {non_numeric:,} non-numeric value(s) in '{col}' will be treated as missing.")

        selected.append(col)
        print(f"  [OK] Added: '{col}'  (Total selected: {len(selected)})")

    print(f"\n  Repeated measures selected ({len(selected)}): {selected}")
    return selected


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               subject_col: str,
               measure_cols: list) -> pd.DataFrame | None:
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    all_cols = [subject_col] + measure_cols
    working = df[all_cols].copy()
    original_rows = len(working)
    print(f"\n  Original rows          : {original_rows:,}")
    print(f"  Subjects (rows)        : {original_rows:,}")
    print(f"  Repeated measures      : {len(measure_cols)}")

    # Drop rows with missing subject ID
    before = len(working)
    working = working.dropna(subset=[subject_col])
    dropped_no_id = before - len(working)
    if dropped_no_id > 0:
        print(f"  Dropped (missing Subject ID)     : {dropped_no_id:,} row(s)")

    # Coerce all measure columns to numeric
    for col in measure_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    # Drop rows with ANY missing or infinite value across repeated measures
    before = len(working)
    working = working.dropna(subset=measure_cols)
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        print(f"  Dropped (NaN in any measure col) : {dropped_nan:,} row(s)")

    before = len(working)
    inf_mask = working[measure_cols].apply(lambda col: np.isinf(col)).any(axis=1)
    working = working[~inf_mask]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        print(f"  Dropped (Inf in any measure col) : {dropped_inf:,} row(s)")

    clean_rows = len(working)
    total_dropped = original_rows - clean_rows
    print(f"\n  Total rows dropped     : {total_dropped:,}")
    print(f"  Subjects for analysis  : {clean_rows:,}")

    if clean_rows == 0:
        print("\n  [ERROR] No valid rows remain after cleaning.")
        print("          Cannot proceed with Friedman Test.")
        return None

    if clean_rows < 3:
        print(f"\n  [WARNING] Only {clean_rows} subject(s) remain after cleaning.")
        print("            Friedman Test results may be highly unreliable.")
        print("            A minimum of 10 subjects is recommended.")

    # Warn on constant columns
    for col in measure_cols:
        if is_effectively_constant(working[col]):
            print(f"  [WARNING] Repeated measure '{col}' has zero variance (all values identical).")

    print(f"\n  [OK] Data is clean and ready for Friedman Test.")
    return working.reset_index(drop=True)


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(df: pd.DataFrame,
                               measure_cols: list) -> dict:
    desc = {}
    for col in measure_cols:
        arr = df[col].to_numpy(dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        desc[col] = {
            "n"       : len(arr),
            "mean"    : float(np.mean(arr)),
            "median"  : float(np.median(arr)),
            "std"     : float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
            "sem"     : float(stats.sem(arr)) if len(arr) > 1 else float("nan"),
            "min"     : float(np.min(arr)),
            "max"     : float(np.max(arr)),
            "q1"      : float(q1),
            "q3"      : float(q3),
            "iqr"     : float(q3 - q1),
            "skewness": float(stats.skew(arr)) if len(arr) > 2 else float("nan"),
            "kurtosis": float(stats.kurtosis(arr)) if len(arr) > 3 else float("nan"),
        }
    return desc


def display_descriptive_stats(desc: dict, measure_cols: list):
    print_section("DESCRIPTIVE STATISTICS")

    print(f"\n  Repeated Measures ({len(measure_cols)}): {measure_cols}\n")

    stat_rows = [
        ("Sample Size (n)",   "n"),
        ("Mean",              "mean"),
        ("Median",            "median"),
        ("Std Deviation",     "std"),
        ("Std Error of Mean", "sem"),
        ("Minimum",           "min"),
        ("Maximum",           "max"),
        ("25th Percentile",   "q1"),
        ("75th Percentile",   "q3"),
        ("IQR (Q3 - Q1)",     "iqr"),
        ("Skewness",          "skewness"),
        ("Kurtosis",          "kurtosis"),
    ]

    col_w     = max(14, max(len(c) for c in measure_cols) + 2)
    label_w   = 28

    hdr = f"  {'Statistic':<{label_w}}"
    for col in measure_cols:
        hdr += f" {str(col)[:col_w]:>{col_w}}"
    print(hdr)

    divider = f"  {'-'*label_w}"
    for _ in measure_cols:
        divider += f" {'-'*col_w}"
    print(divider)

    for display_name, key in stat_rows:
        row = f"  {display_name:<{label_w}}"
        for col in measure_cols:
            row += f" {fmt_val(desc[col][key]):>{col_w}}"
        print(row)


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def check_normality(df: pd.DataFrame, measure_cols: list) -> dict:
    normality = {}
    for col in measure_cols:
        arr = df[col].to_numpy(dtype=float)
        n = len(arr)
        if n < 3:
            normality[col] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": None, "note": "Too few samples (n < 3)"
            }
        elif n > 5000:
            normality[col] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": None,
                "note": "n > 5000: Shapiro-Wilk not computed (large sample)"
            }
        else:
            stat, p = stats.shapiro(arr)
            normality[col] = {
                "stat": float(stat), "p": float(p),
                "normal": p > 0.05, "note": ""
            }
    return normality


def display_assumption_checks(df: pd.DataFrame,
                               measure_cols: list,
                               subject_col: str,
                               normality: dict):
    print_section("ASSUMPTION CHECKS")

    n_subjects = len(df)
    k_measures = len(measure_cols)

    print("\n  [1] Repeated Measures Design")
    print(f"      CONFIRMED: Each row represents one subject measured {k_measures} times.")
    print(f"      Subject ID column : '{subject_col}'")
    print(f"      Subjects          : {n_subjects:,}")
    print(f"      Repeated measures : {k_measures}  ({measure_cols})")

    print("\n  [2] At Least 3 Repeated Measures")
    if k_measures >= 3:
        print(f"      CONFIRMED: {k_measures} repeated measures selected. Minimum requirement met.")
    else:
        print(f"      [FAIL] Only {k_measures} measures found. Friedman Test requires ≥ 3.")

    print("\n  [3] Dependent / Paired Observations")
    print("      ASSUMED: All measurements are taken on the same subjects.")
    print("      The Friedman Test is appropriate for within-subjects designs.")
    print("      It is NOT appropriate when subjects differ between time points.")

    print("\n  [4] Normality Assessment (Shapiro-Wilk Test)")
    print(f"      H0: Data is normally distributed  (p > 0.05 = Normal)")
    print(f"      H1: Data is NOT normally distributed (p ≤ 0.05 = Non-Normal)\n")
    for col, res in normality.items():
        print(f"      Measure: {col}")
        if res["note"]:
            print(f"        Note   : {res['note']}")
        else:
            verdict = "Normal" if res["normal"] is True else "Non-Normal"
            print(f"        W Stat : {res['stat']:.4f}")
            print(f"        p-value: {res['p']:.4f}")
            print(f"        Result : {verdict}")
        print()

    print("  [5] Why Friedman Test?")
    print("      The Friedman Test was selected because:")
    print("      - It is the non-parametric alternative to Repeated Measures ANOVA.")
    print("      - It does not require normality of the repeated measurements.")
    print("      - It is appropriate when data is ordinal or non-normally distributed.")
    print("      - It compares rank-based distributions across multiple time points.")
    print("      - It handles the dependency structure of repeated measurements.")

    print("\n  [6] Missing Data Summary")
    print(f"      Subjects retained for analysis: {n_subjects:,}")
    print("      (Subjects with ANY missing value across repeated measures were excluded.)")
    print("      Listwise deletion was applied to ensure complete cases only.")

    print("\n  [7] Sample Size Note")
    if n_subjects < 10:
        print(f"      [WARNING] Only {n_subjects} subject(s) available.")
        print("               Friedman Test may have low statistical power.")
        print("               Results should be interpreted with caution.")
    elif n_subjects < 30:
        print(f"      [NOTE] {n_subjects} subjects available. Adequate for Friedman Test.")
    else:
        print(f"      [OK] {n_subjects} subjects available. Good statistical power.")


# ======================================================================
# FRIEDMAN TEST
# ======================================================================

def run_friedman_test(df: pd.DataFrame,
                      measure_cols: list) -> dict | None:
    k = len(measure_cols)
    n = len(df)

    arrays = [df[col].to_numpy(dtype=float) for col in measure_cols]

    # Guard: minimum requirements
    if n < 3:
        print(f"  [ERROR] Insufficient subjects (n = {n}). Friedman Test requires n ≥ 3.")
        return None
    if k < 3:
        print(f"  [ERROR] Insufficient repeated measures (k = {k}). Minimum is 3.")
        return None

    try:
        chi2, p_value = friedmanchisquare(*arrays)
    except ValueError as exc:
        print(f"  [ERROR] Friedman Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Friedman Test.\n          Details: {exc}")
        return None

    if math.isnan(chi2) or math.isnan(p_value):
        print("  [ERROR] Friedman Test returned NaN results.")
        print("          This may indicate constant values or insufficient variance.")
        return None

    dof = k - 1
    w = kendalls_w(chi2, n, k)
    effect_label = interpret_kendalls_w(w)

    return {
        "chi2"        : float(chi2),
        "p_value"     : float(p_value),
        "dof"         : dof,
        "n_subjects"  : n,
        "k_measures"  : k,
        "w"           : float(w),
        "effect_label": effect_label,
        "measure_cols": measure_cols,
    }


# ======================================================================
# POST-HOC ANALYSIS
# ======================================================================

def run_posthoc_wilcoxon(df: pd.DataFrame,
                          measure_cols: list) -> list:
    """
    Pairwise Wilcoxon Signed-Rank Tests with Bonferroni correction.
    Returns list of dicts, one per pair.
    """
    pairs = []
    for i in range(len(measure_cols)):
        for j in range(i + 1, len(measure_cols)):
            pairs.append((measure_cols[i], measure_cols[j]))

    n_comparisons = len(pairs)
    results = []

    for col1, col2 in pairs:
        arr1 = df[col1].to_numpy(dtype=float)
        arr2 = df[col2].to_numpy(dtype=float)

        # Wilcoxon requires non-zero differences
        diff = arr1 - arr2
        if np.all(diff == 0):
            results.append({
                "col1"     : col1,
                "col2"     : col2,
                "stat"     : float("nan"),
                "p_raw"    : float("nan"),
                "p_bonf"   : float("nan"),
                "sig_bonf" : False,
                "n_pairs"  : len(arr1),
                "error"    : "All differences are zero; cannot compute Wilcoxon test.",
            })
            continue

        try:
            stat, p_raw = wilcoxon(arr1, arr2, alternative="two-sided",
                                   zero_method="wilcox")
            p_bonf = min(float(p_raw) * n_comparisons, 1.0)
            results.append({
                "col1"     : col1,
                "col2"     : col2,
                "stat"     : float(stat),
                "p_raw"    : float(p_raw),
                "p_bonf"   : float(p_bonf),
                "sig_bonf" : float(p_bonf) < 0.05,
                "n_pairs"  : len(arr1),
                "error"    : None,
            })
        except Exception as exc:
            results.append({
                "col1"     : col1,
                "col2"     : col2,
                "stat"     : float("nan"),
                "p_raw"    : float("nan"),
                "p_bonf"   : float("nan"),
                "sig_bonf" : False,
                "n_pairs"  : len(arr1),
                "error"    : str(exc),
            })

    return results


def display_posthoc(posthoc_results: list):
    n_comparisons = len(posthoc_results)
    bonf_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    print_section("POST-HOC ANALYSIS (Pairwise Wilcoxon Signed-Rank + Bonferroni Correction)")

    print(f"\n  Total pairwise comparisons : {n_comparisons}")
    print(f"  Correction method          : Bonferroni")
    print(f"  Corrected α threshold      : {bonf_alpha:.4f}")
    print(f"  (α = 0.05 / {n_comparisons} comparisons)\n")

    col_m = 16
    col_s = 14
    col_p = 12
    col_b = 16
    col_r = 14

    hdr = (f"  {'Measure 1':<{col_m}} {'Measure 2':<{col_m}}"
           f" {'W Statistic':>{col_s}} {'p-value':>{col_p}}"
           f" {'Bonf. p-value':>{col_b}} {'Significant':>{col_r}}")
    print(hdr)
    print(f"  {'-'*col_m} {'-'*col_m} {'-'*col_s} {'-'*col_p} {'-'*col_b} {'-'*col_r}")

    for res in posthoc_results:
        if res["error"]:
            row = (f"  {str(res['col1']):<{col_m}} {str(res['col2']):<{col_m}}"
                   f" {'ERROR':>{col_s}} {'N/A':>{col_p}}"
                   f" {'N/A':>{col_b}} {'N/A':>{col_r}}")
        else:
            sig_str = "YES *" if res["sig_bonf"] else "no"
            row = (f"  {str(res['col1']):<{col_m}} {str(res['col2']):<{col_m}}"
                   f" {res['stat']:>{col_s},.2f}"
                   f" {res['p_raw']:>{col_p}.4f}"
                   f" {res['p_bonf']:>{col_b}.4f}"
                   f" {sig_str:>{col_r}}")
        print(row)

    sig_pairs = [r for r in posthoc_results if r["sig_bonf"] and not r["error"]]
    print(f"\n  Significant pairs after Bonferroni correction: {len(sig_pairs)} / {n_comparisons}")
    if sig_pairs:
        print(f"\n  Significant pairs:")
        for r in sig_pairs:
            print(f"    {r['col1']} vs {r['col2']}  "
                  f"(W = {r['stat']:.2f}, p_bonf = {r['p_bonf']:.4f})")
    else:
        print("  No pairs reached significance after Bonferroni correction.")


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, desc: dict):
    print_section("FRIEDMAN TEST RESULTS")

    sig = results["p_value"] < 0.05
    measure_cols = results["measure_cols"]

    print(f"\n  Test             : Friedman Chi-Square Test")
    print(f"  Null Hypothesis  : All repeated measures have identical distributions.")
    print(f"  Alt. Hypothesis  : At least one repeated measure distribution differs.")
    print(f"  Significance (α) : 0.05")
    print(f"  Subjects (N)     : {results['n_subjects']:,}")
    print(f"  Measures (k)     : {results['k_measures']}  ({measure_cols})\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Friedman χ² Statistic':<40} {results['chi2']:>20.4f}")
    print(f"  {'Degrees of Freedom (k - 1)':<40} {results['dof']:>20}")
    print(f"  {'p-value':<40} {results['p_value']:>20.4f}")
    print(f"  {'Number of Subjects':<40} {results['n_subjects']:>20,}")
    print(f"  {'Number of Repeated Measures':<40} {results['k_measures']:>20}")
    print(f"  {SEPARATOR2[:68]}")

    print(f"\n  Medians per Repeated Measure:")
    print(f"  {'Measure':<30} {'n':>8} {'Median':>14} {'Mean':>14}")
    print(f"  {'-'*30} {'-'*8} {'-'*14} {'-'*14}")
    for col in measure_cols:
        d = desc[col]
        print(f"  {col:<30} {d['n']:>8,} {d['median']:>14,.4f} {d['mean']:>14,.4f}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result  : {sig_str}")
    print(f"  p-value : {results['p_value']:.4f}  (α = 0.05)")


def display_effect_size(results: dict):
    print_section("EFFECT SIZE")

    w = results["w"]
    print(f"\n  Effect Size Measure : Kendall's W (Coefficient of Concordance)")
    print(f"  Formula             : W = χ²_F / (N × (k - 1))\n")

    print(f"  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'Friedman χ² Statistic':<40} {results['chi2']:>20.4f}")
    print(f"  {'Number of Subjects (N)':<40} {results['n_subjects']:>20,}")
    print(f"  {'Number of Measures (k)':<40} {results['k_measures']:>20}")
    print(f"  {'Kendall\'s W':<40} {w:>20.4f}")
    print(f"  {'Effect Size Interpretation':<40} {results['effect_label']:>20}")

    print(f"\n  Effect Size Benchmarks:")
    print(f"    W < 0.10 → Negligible")
    print(f"    W < 0.30 → Small")
    print(f"    W < 0.50 → Moderate")
    print(f"    W ≥ 0.50 → Large")
    print(f"\n  Kendall's W ranges from 0 (no agreement) to 1 (perfect agreement).")
    print(f"  Higher W indicates stronger concordance across repeated measurements.")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            normality: dict,
                            posthoc: list,
                            subject_col: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    chi2 = results["chi2"]
    p = results["p_value"]
    dof = results["dof"]
    w = results["w"]
    n = results["n_subjects"]
    k = results["k_measures"]
    measure_cols = results["measure_cols"]
    sig = p < 0.05

    print(f"\n  REPEATED MEASURES   : {measure_cols}")
    print(f"  SUBJECT ID COLUMN   : {subject_col}")
    print(f"  NUMBER OF SUBJECTS  : {n:,}")
    print(f"  NUMBER OF MEASURES  : {k}")

    print(f"\n  WHY FRIEDMAN TEST?")
    print(f"  The Friedman Test was selected as the appropriate statistical method")
    print(f"  for this analysis because:")
    print(f"    • It is the non-parametric alternative to Repeated Measures ANOVA.")
    print(f"    • It does not require normally distributed data.")
    print(f"    • It is suitable for ordinal data, skewed distributions, or small samples.")
    print(f"    • It properly accounts for the within-subject dependency structure.")
    print(f"    • It compares rank-based distributions across {k} time points simultaneously.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        print(f"  At least one repeated measurement differs significantly from the others.")
        print(f"  (χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Kendall's W = {w:.4f})")
        medians = {col: desc[col]["median"] for col in measure_cols}
        peak_col = max(medians, key=medians.get)
        trough_col = min(medians, key=medians.get)
        print(f"\n  Highest median : '{peak_col}' (Median = {medians[peak_col]:,.4f})")
        print(f"  Lowest  median : '{trough_col}' (Median = {medians[trough_col]:,.4f})")
    else:
        print(f"  No statistically significant difference detected across repeated measurements.")
        print(f"  (χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Kendall's W = {w:.4f})")
        print(f"\n  The repeated measurements do not show a statistically significant")
        print(f"  change or trend across the {k} time points.")

    print(f"\n  EFFECT SIZE (KENDALL'S W):")
    print(f"  W = {w:.4f} → {results['effect_label']} concordance across repeated measures.")
    if w < 0.10:
        print(f"  The agreement across repeated measurements is negligible.")
    elif w < 0.30:
        print(f"  There is a small but potentially meaningful pattern across time points.")
    elif w < 0.50:
        print(f"  There is moderate concordance across the repeated measurements.")
    else:
        print(f"  There is strong concordance — measurements change consistently across time.")

    if sig and posthoc:
        sig_pairs = [r for r in posthoc if r["sig_bonf"] and not r["error"]]
        print(f"\n  POST-HOC COMMENTARY:")
        if sig_pairs:
            print(f"  After Bonferroni correction, {len(sig_pairs)} pairwise comparison(s) were significant:")
            for r in sig_pairs:
                print(f"    • '{r['col1']}' vs '{r['col2']}'"
                      f"  (W = {r['stat']:.2f}, p_bonf = {r['p_bonf']:.4f})")
        else:
            print(f"  Although the overall Friedman Test is significant, no individual")
            print(f"  pairwise comparison survived Bonferroni correction. This may reflect")
            print(f"  a diffuse pattern of change across time points, or limited power")
            print(f"  in individual comparisons due to the conservative correction.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance across time points does not confirm clinical")
    print(f"  efficacy or meaningful change in patient outcomes. Clinicians should")
    print(f"  consider the magnitude of change (Kendall's W), clinical context,")
    print(f"  minimum clinically important difference (MCID), sample size, and")
    print(f"  patient-specific factors when interpreting these results.")
    print(f"  Repeated measurements should follow a predefined protocol to reduce")
    print(f"  measurement bias and ensure comparability across time points.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Friedman Test assumes all measurements are taken on the same subjects.")
    print(f"  • It does not adjust for covariates or confounding variables.")
    print(f"  • Listwise deletion was used; subjects with any missing value were excluded.")
    print(f"  • This is an observational analysis and does not establish causality.")
    print(f"  • The Bonferroni correction for post-hoc tests is conservative;")
    print(f"    true pairwise differences may not survive correction at small sample sizes.")
    if n < 20:
        print(f"  • Small sample size (N = {n:,}). Power may be limited.")
        print(f"    Interpret results with caution.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 normality: dict,
                 posthoc: list,
                 subject_col: str,
                 dataset_path: str) -> str:
    chi2 = results["chi2"]
    p = results["p_value"]
    dof = results["dof"]
    w = results["w"]
    n = results["n_subjects"]
    k = results["k_measures"]
    measure_cols = results["measure_cols"]
    sig = p < 0.05
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Friedman Test")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    lines.append(f"  Subject ID Column    : {subject_col}")
    lines.append(f"  Repeated Measures    : {measure_cols}")
    lines.append(f"  Number of Subjects   : {n:,}")
    lines.append(f"  Number of Measures   : {k}")
    lines.append("")

    stat_rows = [
        ("Sample Size (n)",   "n"),
        ("Mean",              "mean"),
        ("Median",            "median"),
        ("Std Deviation",     "std"),
        ("Std Error of Mean", "sem"),
        ("Minimum",           "min"),
        ("Maximum",           "max"),
        ("25th Percentile",   "q1"),
        ("75th Percentile",   "q3"),
        ("IQR (Q3 - Q1)",     "iqr"),
        ("Skewness",          "skewness"),
        ("Kurtosis",          "kurtosis"),
    ]
    col_w   = max(14, max(len(c) for c in measure_cols) + 2)
    label_w = 28
    hdr = f"  {'Statistic':<{label_w}}"
    for col in measure_cols:
        hdr += f" {str(col)[:col_w]:>{col_w}}"
    lines.append(hdr)
    div = f"  {'-'*label_w}"
    for _ in measure_cols:
        div += f" {'-'*col_w}"
    lines.append(div)
    for name, key in stat_rows:
        row = f"  {name:<{label_w}}"
        for col in measure_cols:
            row += f" {fmt_val(desc[col][key]):>{col_w}}"
        lines.append(row)

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append(f"  [1] Repeated Measures Design     : CONFIRMED ({n} subjects, {k} measures)")
    lines.append(f"  [2] Minimum 3 Repeated Measures  : CONFIRMED ({k} measures)")
    lines.append(f"  [3] Dependent Observations       : ASSUMED (same subjects)")
    lines.append(f"  [4] Normality (Shapiro-Wilk):")
    for col, res in normality.items():
        if res["note"]:
            lines.append(f"       {col}: {res['note']}")
        else:
            verdict = "Normal" if res["normal"] is True else "Non-Normal"
            lines.append(f"       {col}: W={res['stat']:.4f}, p={res['p']:.4f} → {verdict}")
    lines.append("  [5] Non-parametric alternative to Repeated Measures ANOVA.")
    lines.append("  [6] Listwise deletion applied for missing values.")

    # Test Results
    lines.append("")
    lines.append("  FRIEDMAN TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test                     : Friedman Chi-Square Test")
    lines.append(f"  Null Hypothesis          : All repeated measures have identical distributions")
    lines.append(f"  Friedman χ² Statistic    : {chi2:.4f}")
    lines.append(f"  Degrees of Freedom       : {dof}")
    lines.append(f"  p-value                  : {p:.4f}")
    lines.append(f"  Number of Subjects (N)   : {n:,}")
    lines.append(f"  Number of Measures (k)   : {k}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result                   : {sig_str}")
    lines.append("")
    lines.append(f"  {'Measure':<30} {'n':>8} {'Median':>14} {'Mean':>14}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*14} {'-'*14}")
    for col in measure_cols:
        d = desc[col]
        lines.append(f"  {col:<30} {d['n']:>8,} {d['median']:>14,.4f} {d['mean']:>14,.4f}")

    # Post-Hoc
    lines.append("")
    lines.append("  POST-HOC ANALYSIS (Pairwise Wilcoxon Signed-Rank + Bonferroni Correction)")
    lines.append(SEPARATOR2)
    if not posthoc:
        lines.append("  Post-hoc analysis was not performed.")
        lines.append("  Overall Friedman p ≥ 0.05; pairwise comparisons are not warranted.")
    else:
        n_comp = len(posthoc)
        bonf_alpha = 0.05 / n_comp if n_comp > 0 else 0.05
        lines.append(f"  Total comparisons : {n_comp}")
        lines.append(f"  Correction        : Bonferroni")
        lines.append(f"  Corrected α       : {bonf_alpha:.4f}")
        lines.append("")
        col_m = 16
        col_s = 14
        col_p = 12
        col_b = 16
        col_r = 14
        ph_hdr = (f"  {'Measure 1':<{col_m}} {'Measure 2':<{col_m}}"
                  f" {'W Statistic':>{col_s}} {'p-value':>{col_p}}"
                  f" {'Bonf. p-value':>{col_b}} {'Significant':>{col_r}}")
        lines.append(ph_hdr)
        lines.append(f"  {'-'*col_m} {'-'*col_m} {'-'*col_s} {'-'*col_p} {'-'*col_b} {'-'*col_r}")
        for res in posthoc:
            if res["error"]:
                row = (f"  {str(res['col1']):<{col_m}} {str(res['col2']):<{col_m}}"
                       f" {'ERROR':>{col_s}} {'N/A':>{col_p}}"
                       f" {'N/A':>{col_b}} {'N/A':>{col_r}}")
            else:
                sig_ph = "YES *" if res["sig_bonf"] else "no"
                row = (f"  {str(res['col1']):<{col_m}} {str(res['col2']):<{col_m}}"
                       f" {res['stat']:>{col_s},.2f}"
                       f" {res['p_raw']:>{col_p}.4f}"
                       f" {res['p_bonf']:>{col_b}.4f}"
                       f" {sig_ph:>{col_r}}")
            lines.append(row)

    # Effect Size
    lines.append("")
    lines.append("  EFFECT SIZE")
    lines.append(SEPARATOR2)
    lines.append(f"  Kendall's W                    : {w:.4f}")
    lines.append(f"  Effect Size Interpretation     : {results['effect_label']}")
    lines.append(f"  Formula                        : W = χ²_F / (N × (k - 1))")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append("  At least one repeated measurement differs significantly from the others.")
        lines.append(f"  χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Kendall's W = {w:.4f}")
    else:
        lines.append("  No statistically significant difference detected across repeated measurements.")
        lines.append(f"  χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Kendall's W = {w:.4f}")
    lines.append("")
    lines.append("  The Friedman Test was used as the non-parametric alternative to")
    lines.append("  Repeated Measures ANOVA. It does not require normality and is")
    lines.append("  appropriate for ordinal or non-normally distributed repeated data.")
    lines.append(f"  Kendall's W = {w:.4f} ({results['effect_label']}) indicates the degree")
    lines.append("  of concordance across repeated measurements.")
    lines.append("  Statistical significance does not imply clinical importance.")
    lines.append("  Confounders and covariates have not been adjusted for.")

    lines.append("")
    lines.append(SEPARATOR)
    lines.append("  END OF REPORT")
    lines.append(SEPARATOR)
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str):
    print_section("SAVE REPORT")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n  [OK] Report saved successfully.")
        print(f"       Path: {os.path.abspath(REPORT_FILE)}")
    except PermissionError:
        print(f"  [ERROR] Permission denied. Cannot write to: {REPORT_FILE}")
    except OSError as exc:
        print(f"  [ERROR] Failed to save report.\n          Details: {exc}")


# ======================================================================
# MAIN ORCHESTRATOR
# ======================================================================

def run_friedman_test_module():
    print_header()
    print("  This module performs the Friedman Test.")
    print("  It is the non-parametric alternative to Repeated Measures ANOVA.")
    print("  Suitable for ordinal or non-normally distributed repeated measurements.")
    print("  Requires same subjects measured three or more times (wide-format data).")

    # Step 1: Load CSV
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Subject ID column
    subject_col = select_subject_id_column(df)
    if subject_col is None:
        sys.exit(1)

    # Step 4: Repeated measure columns
    measure_cols = select_repeated_measure_columns(df, subject_col)
    if measure_cols is None or len(measure_cols) < 3:
        print("\n  [FATAL] At least 3 repeated measure columns are required.")
        sys.exit(1)

    # Step 5: Clean data
    clean_df = clean_data(df, subject_col, measure_cols)
    if clean_df is None:
        sys.exit(1)

    # Step 6: Descriptive statistics
    desc = compute_descriptive_stats(clean_df, measure_cols)
    display_descriptive_stats(desc, measure_cols)

    # Step 7: Normality checks
    normality = check_normality(clean_df, measure_cols)

    # Step 8: Assumption checks
    display_assumption_checks(clean_df, measure_cols, subject_col, normality)

    # Step 9: Run Friedman Test
    print_section("RUNNING FRIEDMAN TEST")
    friedman_results = run_friedman_test(clean_df, measure_cols)
    if friedman_results is None:
        print("\n  [FATAL] Friedman Test could not be completed.")
        sys.exit(1)

    display_test_results(friedman_results, desc)

    # Step 10: Post-hoc (only if significant)
    print_section("POST-HOC ANALYSIS")
    if friedman_results["p_value"] < 0.05:
        print(f"\n  p < 0.05 detected. Running pairwise Wilcoxon Signed-Rank tests...")
        posthoc = run_posthoc_wilcoxon(clean_df, measure_cols)
        display_posthoc(posthoc)
    else:
        print(f"\n  p ≥ 0.05. Post-hoc analysis skipped.")
        print(f"  (Overall Friedman Test is not significant;")
        print(f"   pairwise comparisons are not warranted.)")
        posthoc = []

    # Step 11: Effect size
    display_effect_size(friedman_results)

    # Step 12: Medical interpretation
    display_interpretation(friedman_results, desc, normality,
                           posthoc, subject_col)

    # Step 13: Build and save report
    report_text = build_report(friedman_results, desc, normality,
                               posthoc, subject_col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Friedman Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_friedman_test_module()