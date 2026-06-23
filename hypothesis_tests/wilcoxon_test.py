# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Wilcoxon Signed-Rank Test
# File        : hypothesis_tests/wilcoxon_signed_rank_test.py
# Description : Non-parametric alternative to the Paired T-Test.
#               Tests whether two paired measurements differ significantly.
#               Appropriate for ordinal or non-normally distributed data.
#               Uses scipy.stats.wilcoxon().
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import wilcoxon, shapiro, norm

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "wilcoxon_signed_rank_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Wilcoxon Signed-Rank Test
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


def compute_effect_size_r(w_stat: float, n: int) -> float:
    """
    Compute effect size r for Wilcoxon Signed-Rank Test.

        r = Z / sqrt(N)

    The Z-score is derived from the W statistic using the normal
    approximation to the Wilcoxon distribution:

        mean_W  = N(N+1) / 4
        sigma_W = sqrt( N(N+1)(2N+1) / 24 )
        Z       = (W - mean_W) / sigma_W

    Returns r in [-1, 1]. Returns NaN if N < 1.
    """
    if n < 1:
        return float("nan")
    mean_w  = n * (n + 1) / 4.0
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma_w == 0:
        return float("nan")
    z = (w_stat - mean_w) / sigma_w
    r = z / math.sqrt(n)
    return max(-1.0, min(1.0, r))


def interpret_effect_size_r(r: float) -> str:
    """
    Interpret effect size r using standard benchmarks:
        |r| < 0.10 → Negligible
        |r| < 0.30 → Small
        |r| < 0.50 → Moderate
        |r| ≥ 0.50 → Large
    """
    if math.isnan(r):
        return "N/A"
    abs_r = abs(r)
    if abs_r < 0.10:
        return "Negligible"
    elif abs_r < 0.30:
        return "Small"
    elif abs_r < 0.50:
        return "Moderate"
    else:
        return "Large"


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_wilcoxon.csv",
                         n: int = 30,
                         seed: int = 42):
    """
    Generate a synthetic paired dataset for Wilcoxon Signed-Rank Test testing.

    Columns:
        Patient_ID : unique subject identifier
        Before     : baseline measurement (e.g. systolic BP before treatment)
        After      : follow-up measurement (e.g. systolic BP after treatment)
    """
    rng = np.random.default_rng(seed)
    before = rng.normal(loc=145.0, scale=12.0, size=n).clip(100, 200).round(1)
    reduction = rng.normal(loc=10.0, scale=5.0, size=n)
    after = (before - reduction).clip(90, 200).round(1)

    df = pd.DataFrame({
        "Patient_ID": [f"PT{str(i + 1).zfill(3)}" for i in range(n)],
        "Before"    : before,
        "After"     : after,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows: {n}  |  Columns: Patient_ID, Before, After")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    print("\n  NOTE: The Wilcoxon Signed-Rank Test requires PAIRED data.")
    print("        Each row = one subject measured twice.")
    print("        Example: Patient_ID | Before_BP | After_BP")
    print("\n  TIP: To generate a sample dataset for testing, run:")
    print("       from hypothesis_tests.wilcoxon_signed_rank_test import generate_sample_csv")
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
# COLUMN SELECTION & VALIDATION
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

        # HARD ERROR: duplicate Subject IDs
        dupes = df[col].dropna().duplicated()
        n_dupes = dupes.sum()
        if n_dupes > 0:
            dupe_vals = df[col].dropna()[dupes].unique()[:10]
            print(f"\n  [ERROR] Subject ID column '{col}' contains {n_dupes:,} duplicate value(s).")
            print(f"          Duplicate IDs: {list(dupe_vals)}")
            print(f"          The Wilcoxon Signed-Rank Test requires exactly ONE row per subject.")
            print(f"          Duplicate Subject IDs violate the paired observations assumption.")
            print(f"          Please fix the dataset or select a different Subject ID column.")
            continue

        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  [WARNING] {n_missing:,} missing Subject ID(s) detected.")
            print(f"            Rows with missing Subject IDs will be excluded during cleaning.")

        print(f"  [OK] Subject ID column selected: '{col}'")
        print(f"       Unique subjects : {df[col].nunique():,}")
        return col


def select_numeric_column(df: pd.DataFrame,
                           label: str,
                           subject_col: str,
                           exclude_cols: list | None = None) -> str | None:
    exclude_cols = exclude_cols or []
    print(f"\n  The {label} must be a numeric column.")
    print(f"  Example: Before_BP, After_BP, Pain_Score, HbA1c, etc.")

    while True:
        col = input(f"\n  Enter {label} column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue
        if col == subject_col:
            print(f"  [ERROR] Cannot use Subject ID column '{subject_col}' as a measurement.")
            continue
        if col in exclude_cols:
            print(f"  [ERROR] Column '{col}' is already selected. Please choose a different column.")
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        valid_count = coerced.notna().sum()
        if valid_count == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values. Cannot use as measurement.")
            continue
        non_numeric = len(coerced) - valid_count
        if non_numeric > 0:
            print(f"  [WARNING] {non_numeric:,} non-numeric value(s) in '{col}' will be treated as missing.")

        print(f"  [OK] {label} selected: '{col}'  (valid numeric values: {valid_count:,})")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               subject_col: str,
               col1: str,
               col2: str) -> pd.DataFrame | None:
    print_section("STEP 6 | DATA CLEANING & VALIDATION")

    working = df[[subject_col, col1, col2]].copy()
    original_rows = len(working)
    print(f"\n  Original rows          : {original_rows:,}")

    before = len(working)
    working = working.dropna(subset=[subject_col])
    dropped_no_id = before - len(working)
    if dropped_no_id > 0:
        print(f"  Dropped (missing Subject ID)         : {dropped_no_id:,} row(s)")

    working[col1] = pd.to_numeric(working[col1], errors="coerce")
    working[col2] = pd.to_numeric(working[col2], errors="coerce")

    before = len(working)
    working = working.dropna(subset=[col1, col2])
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        print(f"  Dropped (NaN in measurements)        : {dropped_nan:,} row(s)")

    before = len(working)
    inf_mask = (np.isinf(working[col1]) | np.isinf(working[col2]))
    working = working[~inf_mask]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        print(f"  Dropped (Inf in measurements)        : {dropped_inf:,} row(s)")

    clean_rows = len(working)
    total_dropped = original_rows - clean_rows
    print(f"\n  Total rows dropped     : {total_dropped:,}")
    print(f"  Pairs for analysis     : {clean_rows:,}")

    if clean_rows == 0:
        print("\n  [ERROR] No valid paired rows remain after cleaning.")
        print("          Cannot proceed with Wilcoxon Signed-Rank Test.")
        return None

    if clean_rows < 10:
        print(f"\n  [WARNING] Sample size is very small (N = {clean_rows:,}).")
        print("            Wilcoxon normal approximation and effect-size estimates")
        print("            may be unstable. Results should be interpreted with caution.")

    if is_effectively_constant(working[col1]):
        print(f"  [WARNING] Column '{col1}' has zero variance (all values identical).")
    if is_effectively_constant(working[col2]):
        print(f"  [WARNING] Column '{col2}' has zero variance (all values identical).")

    diff = working[col2].to_numpy(dtype=float) - working[col1].to_numpy(dtype=float)

    # HARD STOP: all differences are zero (no change to test)
    if np.allclose(diff, 0):
        print(f"\n  [ERROR] All paired differences between '{col2}' and '{col1}' are zero.")
        print(f"          The Wilcoxon Signed-Rank Test cannot be computed when all")
        print(f"          differences are zero. There is no measurable change to test.")
        print(f"          Please verify your data or column selection.")
        return None

    # WARNING: constant non-zero differences (suspicious data)
    diff_variance = float(np.var(diff, ddof=1)) if len(diff) > 1 else float("nan")
    if not math.isnan(diff_variance) and diff_variance < 1e-10 and not np.allclose(diff, 0):
        print(f"\n  [WARNING] All paired differences are identical (constant non-zero diff).")
        print(f"            Statistical results should be interpreted cautiously.")
        print(f"            This may indicate synthetic, duplicated, or")
        print(f"            data-entry-generated values.")

    n_zero_diff = int(np.sum(np.isclose(diff, 0)))
    if n_zero_diff > 0:
        print(f"  [NOTE] {n_zero_diff:,} pair(s) have zero difference (tied at zero).")
        print(f"         These will be handled by scipy using zero_method='wilcox' (excluded).")

    print(f"\n  [OK] Data is clean and ready for Wilcoxon Signed-Rank Test.")
    return working.reset_index(drop=True)


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(df: pd.DataFrame,
                               col1: str,
                               col2: str) -> dict:
    diff_arr = df[col2].to_numpy(dtype=float) - df[col1].to_numpy(dtype=float)
    diff_series = pd.Series(diff_arr, name=f"{col2} − {col1}")

    desc = {}
    for label, series in [(col1, df[col1]), (col2, df[col2]),
                           (f"{col2} − {col1}", diff_series)]:
        arr = series.to_numpy(dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        n = len(arr)
        desc[label] = {
            "n"       : n,
            "mean"    : float(np.mean(arr)),
            "median"  : float(np.median(arr)),
            "std"     : float(np.std(arr, ddof=1)) if n > 1 else float("nan"),
            "sem"     : float(stats.sem(arr))       if n > 1 else float("nan"),
            "min"     : float(np.min(arr)),
            "max"     : float(np.max(arr)),
            "q1"      : float(q1),
            "q3"      : float(q3),
            "iqr"     : float(q3 - q1),
            "skewness": float(stats.skew(arr))      if n > 2 else float("nan"),
            "kurtosis": float(stats.kurtosis(arr))  if n > 3 else float("nan"),
        }
    return desc


def display_descriptive_stats(desc: dict, col1: str, col2: str):
    print_section("DESCRIPTIVE STATISTICS")

    diff_label = f"{col2} − {col1}"
    display_keys = [col1, col2, diff_label]

    print(f"\n  Measurement 1    : {col1}  (Pre / Before)")
    print(f"  Measurement 2    : {col2}  (Post / After)")
    print(f"  Difference       : {diff_label}  (Post − Pre)\n")

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

    col_w   = max(18, max(len(k) for k in display_keys) + 2)
    label_w = 28

    hdr = f"  {'Statistic':<{label_w}}"
    for k in display_keys:
        hdr += f" {str(k)[:col_w]:>{col_w}}"
    print(hdr)

    divider = f"  {'-'*label_w}"
    for _ in display_keys:
        divider += f" {'-'*col_w}"
    print(divider)

    for display_name, key in stat_rows:
        row = f"  {display_name:<{label_w}}"
        for k in display_keys:
            row += f" {fmt_val(desc[k][key]):>{col_w}}"
        print(row)


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def check_normality_differences(diff_arr: np.ndarray) -> dict:
    n = len(diff_arr)
    if n < 3:
        return {"stat": float("nan"), "p": float("nan"),
                "normal": None, "note": "Too few pairs (n < 3)"}
    if n > 5000:
        return {"stat": float("nan"), "p": float("nan"),
                "normal": None,
                "note": "n > 5000: Shapiro-Wilk not computed (large sample)"}
    stat, p = shapiro(diff_arr)
    return {
        "stat"  : float(stat),
        "p"     : float(p),
        "normal": bool(p > 0.05),   # explicit bool() — safe against numpy.bool_
        "note"  : ""
    }


def display_assumption_checks(df: pd.DataFrame,
                               col1: str,
                               col2: str,
                               subject_col: str,
                               normality: dict):
    print_section("ASSUMPTION CHECKS")

    n_pairs  = len(df)
    diff_arr = df[col2].to_numpy(dtype=float) - df[col1].to_numpy(dtype=float)
    n_zero   = int(np.sum(np.isclose(diff_arr, 0)))

    print("\n  [1] Paired Observations")
    print(f"      CONFIRMED: Each row represents one subject measured twice.")
    print(f"      Subject ID column  : '{subject_col}'")
    print(f"      Measurement 1      : '{col1}'  (Pre / Before)")
    print(f"      Measurement 2      : '{col2}'  (Post / After)")
    print(f"      Valid paired rows  : {n_pairs:,}")

    print("\n  [2] Same Subjects Measured Twice")
    print("      ASSUMED: Both measurements were taken on the same subjects.")
    print("      The Wilcoxon Signed-Rank Test is NOT appropriate for independent groups.")
    print("      For independent groups, use the Mann-Whitney U Test module.")

    print("\n  [3] Difference Distribution Assessment")
    print(f"      Differences computed as: {col2} − {col1}")
    median_diff = float(np.median(diff_arr))
    mean_diff   = float(np.mean(diff_arr))
    n_positive  = int(np.sum(diff_arr > 0))
    n_negative  = int(np.sum(diff_arr < 0))
    print(f"      Mean difference    : {mean_diff:,.4f}")
    print(f"      Median difference  : {median_diff:,.4f}")
    print(f"      Positive changes   : {n_positive:,}  ({n_positive/n_pairs*100:.1f}%)")
    print(f"      Negative changes   : {n_negative:,}  ({n_negative/n_pairs*100:.1f}%)")
    print(f"      Zero differences   : {n_zero:,}  (excluded from ranking by Wilcoxon)")

    print("\n  [4] Normality of Differences (Shapiro-Wilk Test)")
    print(f"      H0: Differences are normally distributed  (p > 0.05 = Normal)")
    print(f"      H1: Differences are NOT normally distributed (p ≤ 0.05 = Non-Normal)\n")
    res = normality
    if res["note"]:
        print(f"      Note   : {res['note']}")
    else:
        verdict = "Normal" if bool(res["normal"]) else "Non-Normal"
        print(f"      W Stat : {res['stat']:.4f}")
        print(f"      p-value: {res['p']:.4f}")
        print(f"      Result : {verdict}")

    print(f"\n      NOTE: Wilcoxon Signed-Rank Test does not require normally distributed")
    print(f"      data. These results are provided for documentation purposes and to")
    print(f"      inform method selection only. If differences are normally distributed,")
    print(f"      a Paired T-Test may be more powerful.")

    print("\n  [5] Why Wilcoxon Signed-Rank Test?")
    print("      The Wilcoxon Signed-Rank Test was selected because:")
    print("      - It is the non-parametric alternative to the Paired T-Test.")
    print("      - It does not require normally distributed data or differences.")
    print("      - It is robust to outliers and skewed distributions.")
    print("      - It tests whether the median difference between paired measurements")
    print("        differs significantly from zero.")
    print("      - It is appropriate for ordinal data or small sample sizes.")

    print("\n  [6] Missing Data Summary")
    print(f"      Pairs retained for analysis : {n_pairs:,}")
    print("      (Pairs with ANY missing or infinite value were excluded using")
    print("       listwise deletion to ensure complete paired observations.)")

    print("\n  [7] Sample Size Note")
    if n_pairs < 10:
        print(f"      [WARNING] Only {n_pairs} pair(s) available.")
        print("               Wilcoxon Test may have very low statistical power.")
        print("               Results should be interpreted with extreme caution.")
    elif n_pairs < 30:
        print(f"      [NOTE] {n_pairs} pairs available. Adequate for Wilcoxon Test.")
    else:
        print(f"      [OK] {n_pairs} pairs available. Good statistical power.")


# ======================================================================
# WILCOXON SIGNED-RANK TEST
# ======================================================================

def run_wilcoxon_test(df: pd.DataFrame,
                      col1: str,
                      col2: str) -> dict | None:
    arr1 = df[col1].to_numpy(dtype=float)
    arr2 = df[col2].to_numpy(dtype=float)
    diff = arr2 - arr1
    n    = len(diff)

    if n < 10:
        print(f"  [WARNING] Sample size is very small (N = {n:,}).")
        print("            Wilcoxon normal approximation and effect-size estimates")
        print("            may be unstable.")

    if np.allclose(diff, 0):
        print("  [ERROR] All paired differences are zero. Wilcoxon Test cannot be computed.")
        return None

    try:
        w_stat, p_value = wilcoxon(arr1, arr2,
                                   alternative="two-sided",
                                   zero_method="wilcox")
    except ValueError as exc:
        print(f"  [ERROR] Wilcoxon Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Wilcoxon Test.\n          Details: {exc}")
        return None

    if math.isnan(w_stat) or math.isnan(p_value):
        print("  [ERROR] Wilcoxon Test returned NaN results.")
        print("          This may indicate all differences are zero or insufficient variance.")
        return None

    n_nonzero = int(np.sum(~np.isclose(diff, 0)))
    r = compute_effect_size_r(float(w_stat), n_nonzero)
    effect_label = interpret_effect_size_r(r)

    mean_w  = n_nonzero * (n_nonzero + 1) / 4.0
    sigma_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
    z_score = (float(w_stat) - mean_w) / sigma_w if sigma_w > 0 else float("nan")

    return {
        "w_stat"      : float(w_stat),
        "p_value"     : float(p_value),
        "z_score"     : float(z_score),
        "n_pairs"     : n,
        "n_nonzero"   : n_nonzero,
        "n_zero_diff" : n - n_nonzero,
        "r"           : float(r),
        "effect_label": effect_label,
        "col1"        : col1,
        "col2"        : col2,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, desc: dict):
    print_section("WILCOXON SIGNED-RANK TEST RESULTS")

    sig = results["p_value"] < 0.05
    col1 = results["col1"]
    col2 = results["col2"]
    diff_label = f"{col2} − {col1}"

    print(f"\n  Test             : Wilcoxon Signed-Rank Test (Two-Sided)")
    print(f"  Measurement 1    : {col1}  (Pre / Before)")
    print(f"  Measurement 2    : {col2}  (Post / After)")
    print(f"  Null Hypothesis  : Median difference between '{col1}' and '{col2}' = 0")
    print(f"  Alt. Hypothesis  : Median difference ≠ 0 (two-sided)")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'W Statistic':<40} {results['w_stat']:>20.4f}")
    print(f"  {'Z Score (approx.)':<40} {fmt_val(results['z_score']):>20}")
    print(f"  {'p-value (two-sided)':<40} {results['p_value']:>20.4f}")
    print(f"  {'Total Paired Observations':<40} {results['n_pairs']:>20,}")
    print(f"  {'Non-zero Differences (used)':<40} {results['n_nonzero']:>20,}")
    print(f"  {'Zero Differences (excluded)':<40} {results['n_zero_diff']:>20,}")
    print(f"  {'Median ' + col1:<40} {desc[col1]['median']:>20,.4f}")
    print(f"  {'Median ' + col2:<40} {desc[col2]['median']:>20,.4f}")
    print(f"  {'Median Difference (' + diff_label + ')':<40} {desc[diff_label]['median']:>20,.4f}")
    print(f"  {'Mean Difference (' + diff_label + ')':<40} {desc[diff_label]['mean']:>20,.4f}")
    print(f"  {SEPARATOR2[:68]}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result  : {sig_str}")
    print(f"  p-value : {results['p_value']:.4f}  (α = 0.05)")


def display_effect_size(results: dict, desc: dict):
    print_section("EFFECT SIZE")

    r          = results["r"]
    col1       = results["col1"]
    col2       = results["col2"]
    diff_label = f"{col2} − {col1}"
    median_diff = desc[diff_label]["median"]

    print(f"\n  Effect Size Measure : r (rank-biserial from Z-score)")
    print(f"  Formula             : r = Z / √N")
    print(f"  Where               : Z = (W − mean_W) / σ_W,  N = non-zero differences\n")

    print(f"  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'W Statistic':<40} {results['w_stat']:>20.4f}")
    print(f"  {'Z Score':<40} {fmt_val(results['z_score']):>20}")
    print(f"  {'N (non-zero differences)':<40} {results['n_nonzero']:>20,}")
    print(f"  {'Effect Size (r)':<40} {fmt_val(r):>20}")
    print(f"  {'Effect Size Interpretation':<40} {results['effect_label']:>20}")

    print(f"\n  Effect Size Benchmarks (Cohen 1988):")
    print(f"    |r| < 0.10 → Negligible")
    print(f"    |r| < 0.30 → Small")
    print(f"    |r| < 0.50 → Moderate")
    print(f"    |r| ≥ 0.50 → Large")

    # Direction determined from median difference, not from sign of r or W
    if median_diff > 0:
        direction_str = f"Measurement 2 ('{col2}') tends to be HIGHER than Measurement 1 ('{col1}')"
    elif median_diff < 0:
        direction_str = f"Measurement 2 ('{col2}') tends to be LOWER than Measurement 1 ('{col1}')"
    else:
        direction_str = "No directional tendency detected (median difference = 0)"

    print(f"\n  Direction : {direction_str}")
    print(f"  (Direction based on median difference = {median_diff:,.4f})")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            normality: dict,
                            col1: str,
                            col2: str,
                            subject_col: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    w    = results["w_stat"]
    p    = results["p_value"]
    r    = results["r"]
    n    = results["n_pairs"]
    sig  = p < 0.05
    diff_label  = f"{col2} − {col1}"
    median_diff = desc[diff_label]["median"]
    mean_diff   = desc[diff_label]["mean"]
    median_pre  = desc[col1]["median"]
    median_post = desc[col2]["median"]

    # Direction from median difference only — never from sign of W or r
    if median_diff > 0:
        direction = "increase"
    elif median_diff < 0:
        direction = "decrease"
    else:
        direction = "no change"

    print(f"\n  MEASUREMENT 1     : {col1}  (Pre / Before)")
    print(f"  MEASUREMENT 2     : {col2}  (Post / After)")
    print(f"  SUBJECT ID        : {subject_col}")
    print(f"  PAIRED SUBJECTS   : {n:,}")

    print(f"\n  WHAT DOES THIS TEST MEASURE?")
    print(f"  The Wilcoxon Signed-Rank Test determines whether two paired measurements")
    print(f"  differ significantly. It ranks the absolute differences between pairs,")
    print(f"  accounting for both the magnitude and direction of change. Unlike the")
    print(f"  Paired T-Test, it does not assume that the differences are normally")
    print(f"  distributed, making it robust for ordinal data, small samples, and")
    print(f"  distributions with outliers.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        print(f"  A statistically significant difference was detected between")
        print(f"  '{col1}' and '{col2}'.")
        print(f"  (W = {w:.4f}, Z ≈ {fmt_val(results['z_score'])}, p = {p:.4f}, r = {fmt_val(r)})")
        print(f"\n  The median value {direction}d from {median_pre:,.4f} ({col1})")
        print(f"  to {median_post:,.4f} ({col2}).")
        print(f"  Median difference ({diff_label}) : {median_diff:,.4f}")
        print(f"  Mean   difference ({diff_label}) : {mean_diff:,.4f}")
    else:
        print(f"  No statistically significant difference was detected between")
        print(f"  '{col1}' and '{col2}'.")
        print(f"  (W = {w:.4f}, Z ≈ {fmt_val(results['z_score'])}, p = {p:.4f}, r = {fmt_val(r)})")
        print(f"\n  The observed difference (Median {diff_label} = {median_diff:,.4f})")
        print(f"  is not statistically significant. The change from '{col1}'")
        print(f"  to '{col2}' may be attributable to chance.")

    print(f"\n  DIRECTION OF CHANGE:")
    if median_diff > 0:
        print(f"  '{col2}' tends to be HIGHER than '{col1}'.")
        print(f"  Median {direction} of {abs(median_diff):,.4f} units observed.")
    elif median_diff < 0:
        print(f"  '{col2}' tends to be LOWER than '{col1}'.")
        print(f"  Median {direction} of {abs(median_diff):,.4f} units observed.")
    else:
        print(f"  No directional change detected. Median difference = 0.")

    print(f"\n  EFFECT SIZE COMMENTARY:")
    abs_r = abs(r) if not math.isnan(r) else float("nan")
    print(f"  Effect size r = {fmt_val(r)} → {results['effect_label']} effect.")
    if not math.isnan(abs_r):
        if abs_r < 0.10:
            print(f"  The practical magnitude of change between measurements is negligible.")
        elif abs_r < 0.30:
            print(f"  There is a small but potentially meaningful change between measurements.")
        elif abs_r < 0.50:
            print(f"  There is a moderate and clinically noteworthy change between measurements.")
        else:
            print(f"  There is a large and practically important change between measurements.")

    print(f"\n  NORMALITY NOTE:")
    if normality["note"]:
        print(f"  {normality['note']}")
    else:
        if bool(normality["normal"]):
            print(f"  Shapiro-Wilk suggests the differences may be normally distributed")
            print(f"  (W = {normality['stat']:.4f}, p = {normality['p']:.4f}).")
            print(f"  A Paired T-Test could also be considered for potentially higher power.")
        else:
            print(f"  Shapiro-Wilk suggests the differences are not normally distributed")
            print(f"  (W = {normality['stat']:.4f}, p = {normality['p']:.4f}).")
            print(f"  The Wilcoxon Signed-Rank Test is the appropriate choice.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance does not automatically imply clinical importance.")
    print(f"  Clinicians should assess whether the magnitude of change (effect size r,")
    print(f"  median difference) meets or exceeds the minimum clinically important")
    print(f"  difference (MCID) for the specific outcome being measured.")
    print(f"  Consider sample size, measurement precision, and patient-level variability")
    print(f"  when translating these findings into clinical practice.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Wilcoxon Signed-Rank Test assumes paired and independent observations.")
    print(f"  • Pairs with zero difference are excluded from ranking (Wilcox method).")
    print(f"  • This test does not adjust for confounding variables or covariates.")
    print(f"  • This is an observational analysis; causation cannot be inferred.")
    print(f"  • Listwise deletion was applied; pairs with any missing value were excluded.")
    if n < 20:
        print(f"  • Small sample size (N = {n:,}). Statistical power may be limited.")
        print(f"    Interpret results with appropriate caution.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 normality: dict,
                 col1: str,
                 col2: str,
                 subject_col: str,
                 dataset_path: str) -> str:
    w    = results["w_stat"]
    p    = results["p_value"]
    r    = results["r"]
    n    = results["n_pairs"]
    sig  = p < 0.05
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diff_label   = f"{col2} − {col1}"
    display_keys = [col1, col2, diff_label]

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Wilcoxon Signed-Rank Test")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    lines.append(f"  Subject ID Column : {subject_col}")
    lines.append(f"  Measurement 1     : {col1}  (Pre / Before)")
    lines.append(f"  Measurement 2     : {col2}  (Post / After)")
    lines.append(f"  Difference        : {diff_label}  (Post − Pre)")
    lines.append(f"  Paired Subjects   : {n:,}")
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
    col_w   = max(18, max(len(k) for k in display_keys) + 2)
    label_w = 28
    hdr = f"  {'Statistic':<{label_w}}"
    for k in display_keys:
        hdr += f" {str(k)[:col_w]:>{col_w}}"
    lines.append(hdr)
    div = f"  {'-'*label_w}"
    for _ in display_keys:
        div += f" {'-'*col_w}"
    lines.append(div)
    for name, key in stat_rows:
        row = f"  {name:<{label_w}}"
        for k in display_keys:
            row += f" {fmt_val(desc[k][key]):>{col_w}}"
        lines.append(row)

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append(f"  [1] Paired Observations           : CONFIRMED ({n:,} pairs)")
    lines.append(f"  [2] Same Subjects Measured Twice  : ASSUMED")
    lines.append(f"  [3] Zero Differences (excluded)   : {results['n_zero_diff']:,}")
    lines.append(f"  [4] Normality of Differences (Shapiro-Wilk):")
    res = normality
    if res["note"]:
        lines.append(f"       Note   : {res['note']}")
    else:
        verdict = "Normal" if bool(res["normal"]) else "Non-Normal"
        lines.append(f"       W Stat : {res['stat']:.4f}")
        lines.append(f"       p-value: {res['p']:.4f}")
        lines.append(f"       Result : {verdict}")
    lines.append("       NOTE: Wilcoxon Signed-Rank Test does not require normally")
    lines.append("       distributed data. These results are provided for documentation")
    lines.append("       purposes and to inform method selection only.")
    lines.append("  [5] Non-parametric alternative to Paired T-Test.")
    lines.append("  [6] Listwise deletion applied for missing values.")

    # Test Results
    lines.append("")
    lines.append("  WILCOXON SIGNED-RANK TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test                       : Wilcoxon Signed-Rank Test (Two-Sided)")
    lines.append(f"  Measurement 1              : {col1}  (Pre / Before)")
    lines.append(f"  Measurement 2              : {col2}  (Post / After)")
    lines.append(f"  W Statistic                : {w:.4f}")
    lines.append(f"  Z Score (approx.)          : {fmt_val(results['z_score'])}")
    lines.append(f"  p-value (two-sided)        : {p:.4f}")
    lines.append(f"  Total Paired Observations  : {n:,}")
    lines.append(f"  Non-zero Differences Used  : {results['n_nonzero']:,}")
    lines.append(f"  Zero Differences Excluded  : {results['n_zero_diff']:,}")
    lines.append(f"  Median {col1:<22}: {desc[col1]['median']:,.4f}")
    lines.append(f"  Median {col2:<22}: {desc[col2]['median']:,.4f}")
    lines.append(f"  Median Difference          : {desc[diff_label]['median']:,.4f}")
    lines.append(f"  Mean   Difference          : {desc[diff_label]['mean']:,.4f}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result                     : {sig_str}")

    # Effect Size
    median_diff = desc[diff_label]["median"]
    if median_diff > 0:
        direction_str = f"Measurement 2 ('{col2}') tends to be HIGHER than Measurement 1 ('{col1}')"
    elif median_diff < 0:
        direction_str = f"Measurement 2 ('{col2}') tends to be LOWER than Measurement 1 ('{col1}')"
    else:
        direction_str = "No directional tendency (median difference = 0)"

    lines.append("")
    lines.append("  EFFECT SIZE")
    lines.append(SEPARATOR2)
    lines.append(f"  Effect Size (r)                : {fmt_val(r)}")
    lines.append(f"  Effect Size Interpretation     : {results['effect_label']}")
    lines.append(f"  Formula                        : r = Z / √N")
    lines.append(f"  Z Score                        : {fmt_val(results['z_score'])}")
    lines.append(f"  N (non-zero differences)       : {results['n_nonzero']:,}")
    lines.append(f"  Direction                      : {direction_str}")
    lines.append(f"  (Direction based on median difference = {median_diff:,.4f})")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append("  A statistically significant difference was detected between paired measurements.")
        lines.append(f"  W = {w:.4f}, Z ≈ {fmt_val(results['z_score'])}, p = {p:.4f}, r = {fmt_val(r)}")
    else:
        lines.append("  No statistically significant difference detected between paired measurements.")
        lines.append(f"  W = {w:.4f}, Z ≈ {fmt_val(results['z_score'])}, p = {p:.4f}, r = {fmt_val(r)}")
    lines.append("")
    lines.append("  The Wilcoxon Signed-Rank Test was used as the non-parametric alternative")
    lines.append("  to the Paired T-Test. It does not require normally distributed differences")
    lines.append("  and is robust to outliers and skewed distributions.")
    lines.append(f"  Median difference ({diff_label}) = {desc[diff_label]['median']:,.4f}")
    lines.append(f"  Effect size r = {fmt_val(r)} ({results['effect_label']}).")
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

def run_wilcoxon_signed_rank_test():
    print_header()
    print("  This module performs the Wilcoxon Signed-Rank Test.")
    print("  It is the non-parametric alternative to the Paired T-Test.")
    print("  Suitable for paired, ordinal, or non-normally distributed data.")
    print("  Each subject must appear exactly once, measured at two time points.")

    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    display_columns(df)

    subject_col = select_subject_id_column(df)
    if subject_col is None:
        sys.exit(1)

    print_section("STEP 4 | SELECT MEASUREMENT COLUMNS")
    col1 = select_numeric_column(df, "Measurement 1 (Pre / Before)", subject_col)
    if col1 is None:
        sys.exit(1)

    col2 = select_numeric_column(df, "Measurement 2 (Post / After)",
                                  subject_col, exclude_cols=[col1])
    if col2 is None:
        sys.exit(1)

    clean_df = clean_data(df, subject_col, col1, col2)
    if clean_df is None:
        sys.exit(1)

    desc = compute_descriptive_stats(clean_df, col1, col2)
    display_descriptive_stats(desc, col1, col2)

    diff_arr = clean_df[col2].to_numpy(dtype=float) - clean_df[col1].to_numpy(dtype=float)
    normality = check_normality_differences(diff_arr)

    display_assumption_checks(clean_df, col1, col2, subject_col, normality)

    print_section("RUNNING WILCOXON SIGNED-RANK TEST")
    results = run_wilcoxon_test(clean_df, col1, col2)
    if results is None:
        print("\n  [FATAL] Wilcoxon Signed-Rank Test could not be completed.")
        sys.exit(1)

    display_test_results(results, desc)
    display_effect_size(results, desc)
    display_interpretation(results, desc, normality, col1, col2, subject_col)

    report_text = build_report(results, desc, normality, col1, col2, subject_col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Wilcoxon Signed-Rank Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_wilcoxon_signed_rank_test()