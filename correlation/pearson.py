# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Pearson Correlation
# File        : correlation/pearson.py
# Description : Measures the strength and direction of a linear
#               relationship between two continuous variables.
#               Uses scipy.stats.pearsonr().
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "pearson_result.txt")
MIN_SAMPLE_SIZE    = 3
LARGE_SAMPLE_N     = 200

# ======================================================================
# REPORT BUFFER
# ======================================================================

class ReportBuffer:
    """Accumulates all output so console and saved report are identical."""
    def __init__(self):
        self._lines: list[str] = []

    def write(self, text: str = ""):
        print(text)
        self._lines.append(text)

    def get(self) -> str:
        return "\n".join(self._lines)


_buf = ReportBuffer()


def rprint(text: str = ""):
    _buf.write(text)


# ======================================================================
# UTILITIES
# ======================================================================

HEADER = """
======================================================================
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Pearson Correlation
======================================================================
"""


def print_header():
    rprint(HEADER)


def print_section(title: str):
    rprint(f"\n{SEPARATOR2}")
    rprint(f"  {title}")
    rprint(SEPARATOR2)


def fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        return f"{val:,.{decimals}f}"
    return str(val)


def is_effectively_constant(arr: np.ndarray, tol: float = 1e-10) -> bool:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return True
    return bool((np.max(finite) - np.min(finite)) < tol)


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    while True:
        path = input("\n  Enter path to CSV file: ").strip()
        if not path:
            print("  [ERROR] No path entered. Please try again.")
            continue
        if not path.lower().endswith(".csv"):
            print("  [WARNING] File does not have a .csv extension. Attempting to load anyway.")
        if not os.path.isfile(path):
            print(f"  [ERROR] File not found: '{path}'. Please check the path.")
            continue
        return path


def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print("  [ERROR] CSV file is empty.")
        return None
    except pd.errors.ParserError as exc:
        print(f"  [ERROR] Failed to parse CSV file.\n          {exc}")
        return None
    except PermissionError:
        print("  [ERROR] Permission denied.")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error loading file.\n          {exc}")
        return None

    if df.empty:
        print("  [ERROR] Dataset has no rows.")
        return None

    rprint(f"\n  [OK] Dataset loaded.")
    rprint(f"       Rows    : {df.shape[0]:,}")
    rprint(f"       Columns : {df.shape[1]:,}")
    return df


def display_columns(df: pd.DataFrame):
    print_section("STEP 2 | AVAILABLE COLUMNS")
    rprint(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15}")
    rprint(f"  {'-'*5} {'-'*40} {'-'*15}")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        rprint(f"  {idx:<5} {col:<40} {str(dtype):<15}")


# ======================================================================
# VARIABLE SELECTION
# ======================================================================

def select_variable(df: pd.DataFrame,
                    label: str,
                    exclude: str | None = None) -> str | None:
    rprint(f"\n  Select {label} — must be a continuous numeric variable.")
    while True:
        col = input(f"  Enter column name for {label}: ").strip()
        if not col:
            print("  [ERROR] No column entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            continue
        if exclude and col == exclude:
            print(f"  [ERROR] Variable 2 cannot be the same as Variable 1 ('{exclude}').")
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() == 0:
            print(f"  [ERROR] Column '{col}' has no numeric values.")
            continue
        non_num = (coerced.isna().sum() - df[col].isna().sum())
        if non_num > 0:
            print(f"  [WARNING] {non_num:,} non-numeric value(s) in '{col}' "
                  f"will be treated as missing.")
        rprint(f"  [OK] {label}: '{col}'")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               col1: str,
               col2: str) -> tuple[np.ndarray, np.ndarray] | None:
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    working = df[[col1, col2]].copy()
    original_n = len(working)
    rprint(f"\n  Original rows : {original_n:,}")

    working[col1] = pd.to_numeric(working[col1], errors="coerce")
    working[col2] = pd.to_numeric(working[col2], errors="coerce")

    before = len(working)
    working = working.dropna(subset=[col1, col2])
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        rprint(f"  Dropped (NaN)      : {dropped_nan:,} row(s)")

    before = len(working)
    inf_mask = np.isinf(working[col1]) | np.isinf(working[col2])
    working  = working[~inf_mask]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        rprint(f"  Dropped (Infinite) : {dropped_inf:,} row(s)")

    n = len(working)
    rprint(f"\n  Total dropped  : {original_n - n:,}")
    rprint(f"  Valid pairs    : {n:,}")

    if n == 0:
        rprint("\n  [ERROR] No valid paired observations remain.")
        return None

    if n < MIN_SAMPLE_SIZE:
        rprint(f"\n  [ERROR] Only {n} valid observation(s) remain.")
        rprint(f"          Pearson correlation requires n ≥ {MIN_SAMPLE_SIZE}.")
        return None

    arr1 = working[col1].to_numpy(dtype=float)
    arr2 = working[col2].to_numpy(dtype=float)

    if is_effectively_constant(arr1):
        rprint(f"\n  [ERROR] Variable '{col1}' is constant (zero variance).")
        rprint("          Pearson correlation is undefined for a constant variable.")
        return None

    if is_effectively_constant(arr2):
        rprint(f"\n  [ERROR] Variable '{col2}' is constant (zero variance).")
        rprint("          Pearson correlation is undefined for a constant variable.")
        return None

    if n < 10:
        rprint(f"\n  [WARNING] Small sample (n = {n}). Pearson r may be unstable.")
    elif n < 30:
        rprint(f"\n  [WARNING] Moderate sample (n = {n}). Interpret with caution.")

    rprint(f"\n  [OK] Data is clean and ready for analysis.")
    return arr1, arr2


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_desc(arr: np.ndarray) -> dict:
    return {
        "n"     : len(arr),
        "mean"  : float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std"   : float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "min"   : float(np.min(arr)),
        "max"   : float(np.max(arr)),
    }


def display_descriptive_stats(arr1: np.ndarray, arr2: np.ndarray,
                               col1: str, col2: str):
    print_section("DESCRIPTIVE STATISTICS")

    d1 = compute_desc(arr1)
    d2 = compute_desc(arr2)

    lw = 24
    cw = 18
    rprint(f"\n  {'Statistic':<{lw}} {col1:>{cw}} {col2:>{cw}}")
    rprint(f"  {'-'*lw} {'-'*cw} {'-'*cw}")

    rows = [
        ("Sample Size (n)", "n"),
        ("Mean",            "mean"),
        ("Median",          "median"),
        ("Std Deviation",   "std"),
        ("Minimum",         "min"),
        ("Maximum",         "max"),
    ]
    for label, key in rows:
        rprint(f"  {label:<{lw}} {fmt(d1[key]):>{cw}} {fmt(d2[key]):>{cw}}")


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def display_assumption_checks(arr1: np.ndarray, arr2: np.ndarray,
                               col1: str, col2: str):
    print_section("ASSUMPTION CHECKS")

    n = len(arr1)

    rprint("\n  [1] Continuous Variables")
    rprint(f"      ASSUMED: Both '{col1}' and '{col2}' are continuous and numeric.")
    rprint("      Pearson correlation is appropriate for continuous data.")
    rprint("      For ordinal variables, use Spearman or Kendall's Tau instead.")

    rprint("\n  [2] Independence of Observations")
    rprint("      ASSUMED: Each observation is independent.")
    rprint("      Pearson correlation is not appropriate for repeated measures")
    rprint("      or clustered data without adjustment.")

    rprint("\n  [3] Linear Relationship")
    rprint("      Pearson r measures the strength of a LINEAR relationship only.")
    rprint("      A non-significant r does not rule out a non-linear association.")
    rprint("      Inspect a scatterplot to verify the relationship is approximately linear.")

    rprint("\n  [4] Outlier Sensitivity")
    # Flag potential outliers using IQR method on each variable
    def count_outliers(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-10:
            return 0
        return int(np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)))

    n_out1 = count_outliers(arr1)
    n_out2 = count_outliers(arr2)

    if n_out1 > 0 or n_out2 > 0:
        rprint(f"      [WARNING] Potential outliers detected:")
        if n_out1 > 0:
            rprint(f"        '{col1}': {n_out1} potential outlier(s) (IQR rule).")
        if n_out2 > 0:
            rprint(f"        '{col2}': {n_out2} potential outlier(s) (IQR rule).")
        rprint("      Pearson r is sensitive to outliers. A single extreme observation")
        rprint("      can substantially inflate or deflate the correlation.")
        rprint("      Consider Spearman correlation as a robust alternative.")
    else:
        rprint("      No obvious outliers detected by IQR rule.")
        rprint("      Pearson r remains sensitive to extreme values.")
        rprint("      Visual inspection of a scatterplot is recommended.")

    rprint("\n  [5] Large Sample Interpretation")
    if n >= LARGE_SAMPLE_N:
        rprint(f"      [NOTE] Large sample (n = {n:,}).")
        rprint("      With large samples, even trivially small correlations")
        rprint("      can reach statistical significance (p < 0.05).")
        rprint("      Assess the magnitude of r, not the p-value alone,")
        rprint("      when judging practical importance.")
    else:
        rprint(f"      Sample size n = {n:,}.")
        rprint("      With smaller samples, the test has reduced power.")
        rprint("      A non-significant result does not confirm the absence of")
        rprint("      a true correlation.")


# ======================================================================
# PEARSON CORRELATION
# ======================================================================

def run_pearson(arr1: np.ndarray,
                arr2: np.ndarray) -> dict | None:
    """
    Compute Pearson correlation using scipy.stats.pearsonr().

    Returns r, p-value, and n. Returns None on failure.

    The p-value tests H₀: ρ = 0 against H₁: ρ ≠ 0 (two-sided).
    It is based on the t-distribution with n−2 degrees of freedom:
        t = r √(n−2) / √(1−r²)
    """
    n = len(arr1)
    try:
        r, p = pearsonr(arr1, arr2)
    except Exception as exc:
        rprint(f"  [ERROR] Pearson correlation failed: {exc}")
        return None

    if math.isnan(r) or math.isnan(p):
        rprint("  [ERROR] Pearson correlation returned NaN.")
        rprint("          This may indicate a constant variable or insufficient data.")
        return None

    return {
        "r"  : float(r),
        "p"  : float(p),
        "n"  : n,
        "r2" : float(r ** 2),
    }


# ======================================================================
# DISPLAY RESULTS
# ======================================================================

def interpret_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r < 0.20:
        return "Very Weak"
    elif abs_r < 0.40:
        return "Weak"
    elif abs_r < 0.60:
        return "Moderate"
    elif abs_r < 0.80:
        return "Strong"
    else:
        return "Very Strong"


def interpret_direction(r: float) -> str:
    if r > 0.0:
        return "Positive"
    elif r < 0.0:
        return "Negative"
    else:
        return "None"


def display_results(res: dict, col1: str, col2: str):
    print_section("PEARSON CORRELATION RESULTS")

    r         = res["r"]
    p         = res["p"]
    n         = res["n"]
    r2        = res["r2"]
    sig       = p < 0.05
    strength  = interpret_strength(r)
    direction = interpret_direction(r)

    rprint(f"\n  Variable 1       : {col1}")
    rprint(f"  Variable 2       : {col2}")
    rprint(f"  H₀               : No linear correlation (ρ = 0)")
    rprint(f"  H₁               : A linear correlation exists (ρ ≠ 0)")
    rprint(f"  Significance (α) : 0.05\n")

    rprint(f"  {SEPARATOR2[:68]}")
    rprint(f"  {'Statistic':<40} {'Value':>20}")
    rprint(f"  {SEPARATOR2[:68]}")
    rprint(f"  {'Pearson r':<40} {r:>20.4f}")
    rprint(f"  {'r² (Coefficient of Determination)':<40} {r2:>20.4f}")
    rprint(f"  {'p-value (two-sided)':<40} {p:>20.4f}")
    rprint(f"  {'Sample Size (n)':<40} {n:>20,}")
    rprint(f"  {'Direction':<40} {direction:>20}")
    rprint(f"  {'Strength':<40} {strength:>20}")
    rprint(f"  {SEPARATOR2[:68]}")

    result_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    rprint(f"\n  Result  : {result_str}")
    rprint(f"  p-value : {p:.4f}  (α = 0.05)")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(res: dict, col1: str, col2: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    r         = res["r"]
    p         = res["p"]
    n         = res["n"]
    r2        = res["r2"]
    sig       = p < 0.05
    strength  = interpret_strength(r)
    direction = interpret_direction(r)
    abs_r     = abs(r)

    rprint(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        rprint(f"  A statistically significant {direction.lower()} linear correlation")
        rprint(f"  was detected between '{col1}' and '{col2}'.")
        rprint(f"  (r = {r:.4f}, p = {p:.4f}, n = {n:,})")
        rprint(f"\n  As '{col1}' increases, '{col2}' tends to "
               f"{'increase' if r > 0 else 'decrease'}.")
    else:
        rprint(f"  No statistically significant linear correlation was detected")
        rprint(f"  between '{col1}' and '{col2}'.")
        rprint(f"  (r = {r:.4f}, p = {p:.4f}, n = {n:,})")
        rprint(f"\n  The observed correlation may be due to chance at this sample size.")

    rprint(f"\n  EFFECT MAGNITUDE:")
    rprint(f"  r = {r:.4f} indicates a {strength.lower()} {direction.lower()} correlation.")
    rprint(f"  r² = {r2:.4f}: {r2*100:.1f}% of the variance in one variable is")
    rprint(f"  linearly explained by the other.")

    if abs_r < 0.20:
        rprint(f"  The practical association between these variables is negligible.")
    elif abs_r < 0.40:
        rprint(f"  The practical association is weak and may have limited clinical utility.")
    elif abs_r < 0.60:
        rprint(f"  The association is moderate and may be clinically relevant depending on context.")
    elif abs_r < 0.80:
        rprint(f"  The association is strong and is likely to be clinically meaningful.")
    else:
        rprint(f"  The association is very strong and is likely clinically important.")

    rprint(f"\n  STATISTICAL vs PRACTICAL SIGNIFICANCE:")
    rprint(f"  A statistically significant p-value indicates the correlation is")
    rprint(f"  unlikely to be zero in the population. It does NOT indicate")
    rprint(f"  the correlation is clinically meaningful or large.")
    if n >= LARGE_SAMPLE_N:
        rprint(f"\n  LARGE SAMPLE NOTE (n = {n:,}):")
        rprint(f"  With large samples, even very small correlations (e.g. r = 0.05)")
        rprint(f"  can reach statistical significance. Focus on the magnitude")
        rprint(f"  of r and its clinical relevance, not the p-value alone.")


def display_limitations(col1: str, col2: str):
    print_section("LIMITATIONS")

    rprint(f"\n  • CORRELATION ≠ CAUSATION.")
    rprint(f"    A significant Pearson r between '{col1}' and '{col2}' does not")
    rprint(f"    establish that one causes the other.")
    rprint(f"\n  • LINEAR RELATIONSHIPS ONLY.")
    rprint(f"    Pearson r detects only linear associations. A strong non-linear")
    rprint(f"    relationship (e.g. quadratic, exponential) may produce r ≈ 0.")
    rprint(f"\n  • SENSITIVE TO OUTLIERS.")
    rprint(f"    A single extreme observation can substantially change the value of r.")
    rprint(f"    Spearman correlation is more robust to outliers.")
    rprint(f"\n  • CONTINUOUS VARIABLES ASSUMED.")
    rprint(f"    Pearson r is designed for continuous data. Results may be")
    rprint(f"    misleading for ordinal or discrete variables with few categories.")
    rprint(f"\n  • LARGE SAMPLES.")
    rprint(f"    In large samples, statistically significant correlations may be")
    rprint(f"    too small to be practically meaningful. Always report r alongside p.")


# ======================================================================
# REPORT SAVE
# ======================================================================

def save_report():
    print_section("SAVE REPORT")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(_buf.get())
        rprint(f"\n  [OK] Report saved.")
        rprint(f"       Path: {os.path.abspath(REPORT_FILE)}")
    except PermissionError:
        print(f"  [ERROR] Permission denied: {REPORT_FILE}")
    except OSError as exc:
        print(f"  [ERROR] {exc}")


# ======================================================================
# MAIN ORCHESTRATOR
# ======================================================================

def run_pearson_correlation():
    print_header()
    rprint("  This module computes Pearson Correlation.")
    rprint("  It measures the strength and direction of a linear relationship")
    rprint("  between two continuous variables.")

    # Step 1: Load
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3 & 4: Select variables
    print_section("STEP 3 | SELECT VARIABLES")
    col1 = select_variable(df, "Variable 1")
    if col1 is None:
        sys.exit(1)
    col2 = select_variable(df, "Variable 2", exclude=col1)
    if col2 is None:
        sys.exit(1)

    # Step 5: Clean data
    result = clean_data(df, col1, col2)
    if result is None:
        sys.exit(1)
    arr1, arr2 = result

    # Step 6: Descriptive statistics
    display_descriptive_stats(arr1, arr2, col1, col2)

    # Step 7: Assumption checks
    display_assumption_checks(arr1, arr2, col1, col2)

    # Step 8: Run Pearson
    print_section("STEP 6 | RUNNING PEARSON CORRELATION")
    res = run_pearson(arr1, arr2)
    if res is None:
        sys.exit(1)

    # Step 9: Results
    display_results(res, col1, col2)

    # Step 10: Interpretation
    display_interpretation(res, col1, col2)

    # Step 11: Limitations
    display_limitations(col1, col2)

    # Step 12: Save report
    save_report()

    rprint(f"\n{SEPARATOR}")
    rprint(f"  VM-WIT-STATS | Pearson Correlation — Analysis Complete")
    rprint(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_pearson_correlation()