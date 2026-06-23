# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Anderson-Darling Normality Test
# File        : normality_tests/anderson_test.py
# Description : Tests whether a continuous variable follows a normal
#               distribution using the Anderson-Darling test.
#               Uses scipy.stats.anderson(dist='norm').
#
# STATISTICAL NOTES:
#   1. Unlike Shapiro-Wilk, anderson() does NOT impose a 5000-sample
#      limit. It is valid for large datasets, but sensitivity increases
#      with n — trivial deviations become detectable at large n.
#
#   2. The primary decision basis is the CRITICAL VALUE TABLE at each
#      significance level. scipy also provides an interpolated p-value
#      (scipy >= 1.17, method='interpolate'), but this p-value is
#      bounded by the table range [0.01, 0.15]. Values outside this
#      range are reported as ≤ 0.01 or ≥ 0.15 respectively.
#      This module reports the p-value with its bounded interpretation
#      documented explicitly.
#
#   3. The anderson() API does NOT return a single definitive p-value
#      in the traditional sense. The critical value table comparison is
#      the statistically correct decision mechanism. The p-value from
#      method='interpolate' is a convenience approximation only.
#
#   4. All boolean normality flags use bool() coercion to prevent
#      numpy.bool_ identity issues (numpy.bool_(True) is True → False).
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import anderson

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "anderson_test_result.txt")

ANDERSON_MIN_N = 3    # Hard minimum for meaningful AD statistic

# Standard significance levels returned by scipy.stats.anderson
# Listed from LEAST to MOST stringent (15% → 1%)
SIGNIFICANCE_LEVELS = [15.0, 10.0, 5.0, 2.5, 1.0]

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Anderson-Darling Normality Test
{SEPARATOR}
"""


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_header() -> None:
    print(HEADER)


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR2}")
    print(f"  {title}")
    print(SEPARATOR2)


def fmt_val(val) -> str:
    """Format a value for display, handling None, NaN, and Inf."""
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


def is_effectively_constant(arr: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Return True if all values in arr are effectively identical.

    Checks range (max - min) rather than std, because std can be
    non-zero due to floating-point rounding even for constant arrays.
    Anderson-Darling returns NaN for constant data — this guard
    prevents misleading 'test failed' messages from reaching the user.
    """
    if len(arr) == 0:
        return True
    return bool(float(np.max(arr)) - float(np.min(arr)) < tol)


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_anderson.csv",
                         n: int = 120,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic clinical dataset for Anderson-Darling testing.

    Columns:
        Patient_ID   : unique subject identifier
        systolic_bp  : approximately normally distributed
        hba1c        : right-skewed (non-normal)
        cholesterol  : approximately normal with mild skew
        creatinine   : right-skewed (log-normal)

    Usage:
        from normality_tests.anderson_test import generate_sample_csv
        generate_sample_csv("data/sample_anderson.csv")
    """
    rng = np.random.default_rng(seed)

    systolic_bp = rng.normal(loc=125.0, scale=15.0, size=n).clip(80, 200).round(1)
    hba1c       = (rng.exponential(scale=1.5, size=n) + 5.0).clip(4.0, 15.0).round(1)
    cholesterol = rng.normal(loc=195.0, scale=35.0, size=n).clip(100, 350).round(0)
    creatinine  = rng.lognormal(mean=0.1, sigma=0.4, size=n).clip(0.5, 10.0).round(2)

    df = pd.DataFrame({
        "Patient_ID" : [f"PT{str(i + 1).zfill(4)}" for i in range(n)],
        "systolic_bp": systolic_bp,
        "hba1c"      : hba1c,
        "cholesterol": cholesterol,
        "creatinine" : creatinine,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows    : {n}")
    print(f"       Columns : Patient_ID, systolic_bp (normal), hba1c (skewed),")
    print(f"                 cholesterol (near-normal), creatinine (log-normal)")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    """Prompt user for CSV file path with validation loop."""
    print_section("STEP 1 | LOAD DATASET")
    print("\n  NOTE: The Anderson-Darling Test evaluates a single continuous variable.")
    print("        Each row should represent one independent observation.")
    print("        Unlike Shapiro-Wilk, there is no hard upper sample size limit.")
    print("\n  TIP: To generate a sample dataset for testing, run:")
    print("       from normality_tests.anderson_test import generate_sample_csv")
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
    """
    Load a CSV file safely with comprehensive exception handling.
    Returns None on any failure so the caller can exit gracefully.
    """
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


def display_columns(df: pd.DataFrame) -> None:
    """Print a numbered table of all column names and dtypes."""
    print_section("STEP 2 | AVAILABLE COLUMNS")
    print(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15} {'Non-Null':>10} {'Null':>8}")
    print(f"  {'-'*5} {'-'*40} {'-'*15} {'-'*10} {'-'*8}")
    total = len(df)
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        non_null = int(df[col].notna().sum())
        null_cnt = total - non_null
        print(f"  {idx:<5} {col:<40} {str(dtype):<15} {non_null:>10,} {null_cnt:>8,}")
    print(f"\n  Total rows: {total:,}   Total columns: {len(df.columns):,}")


# ======================================================================
# COLUMN SELECTION & VALIDATION
# ======================================================================

def select_numeric_column(df: pd.DataFrame) -> str | None:
    """
    Prompt the user to select one numeric column for normality testing.

    Validates existence and numeric content at selection time.
    Hard validation (n >= 3, non-constant) occurs after cleaning.
    """
    print_section("STEP 3 | SELECT COLUMN FOR NORMALITY TEST")
    print("\n  Select ONE continuous numeric column to test for normality.")
    print("  Examples: 'systolic_bp', 'hba1c', 'age', 'charges', 'creatinine'")

    while True:
        col = input("\n  Enter column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

        coerced   = pd.to_numeric(df[col], errors="coerce")
        valid_n   = int(coerced.notna().sum())
        total_n   = len(coerced)
        non_num   = total_n - valid_n

        if valid_n == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values.")
            print(f"          Please select a continuous numeric column.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [WARNING] Column '{col}' has dtype '{df[col].dtype}'.")
            print(f"            {valid_n:,} of {total_n:,} values are numeric and will be used.")
            print(f"            {non_num:,} non-numeric value(s) will be treated as missing.")
        elif non_num > 0:
            print(f"  [WARNING] {non_num:,} non-numeric/missing value(s) detected in '{col}'.")
            print(f"            These will be excluded from analysis.")

        print(f"  [OK] Column selected: '{col}'  (valid numeric values: {valid_n:,})")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_column(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """
    Extract, clean, and validate a single numeric column.

    Steps:
        1. Coerce to numeric (non-parseable → NaN).
        2. Drop NaN values (missing).
        3. Drop ±Inf values.
        4. Report counts at each step.
        5. Hard stop if n < ANDERSON_MIN_N (3).
        6. Hard stop if column is effectively constant:
           anderson() returns NaN for constant data, which would produce
           a misleading 'test failed' message without this guard.

    Returns:
        Clean numpy float64 array, or None if analysis cannot proceed.
    """
    print_section("STEP 4 | DATA CLEANING & VALIDATION")

    series     = pd.to_numeric(df[col], errors="coerce")
    original_n = len(series)
    print(f"\n  Column          : '{col}'")
    print(f"  Original rows   : {original_n:,}")

    nan_count = int(series.isna().sum())
    series    = series.dropna()
    if nan_count > 0:
        print(f"  Dropped (NaN)   : {nan_count:,} row(s)")

    arr       = series.to_numpy(dtype=float)
    inf_mask  = np.isinf(arr)
    inf_count = int(inf_mask.sum())
    arr       = arr[~inf_mask]
    if inf_count > 0:
        print(f"  Dropped (Inf)   : {inf_count:,} row(s)")

    clean_n    = len(arr)
    total_drop = original_n - clean_n
    print(f"  Total dropped   : {total_drop:,}")
    print(f"  Usable values   : {clean_n:,}")

    if clean_n < ANDERSON_MIN_N:
        print(f"\n  [ERROR] Only {clean_n:,} usable value(s) remain after cleaning.")
        print(f"          Anderson-Darling Test requires at least {ANDERSON_MIN_N} observations.")
        print(f"          Cannot proceed with normality testing.")
        return None

    if is_effectively_constant(arr):
        print(f"\n  [ERROR] Column '{col}' is effectively constant.")
        print(f"          All {clean_n:,} values are identical (or differ by < 1e-10).")
        print(f"          Normality is undefined for a constant variable.")
        print(f"          Note: scipy.stats.anderson() returns NaN for constant data.")
        print(f"          Analysis stopped to prevent misleading output.")
        return None

    if clean_n < 20:
        print(f"\n  [WARNING] Small sample (n = {clean_n:,}).")
        print(f"            Anderson-Darling has low power for small samples.")
        print(f"            A non-significant result does not confirm normality.")
    elif clean_n > 200:
        print(f"\n  [NOTE] Large sample (n = {clean_n:,}).")
        print(f"         Anderson-Darling is sensitive at large n. Even minor")
        print(f"         deviations from normality may yield a significant result.")
        print(f"         Consider clinical context when interpreting the outcome.")
    else:
        print(f"\n  [OK] Data cleaned and validated. Ready for Anderson-Darling Test.")

    return arr


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(arr: np.ndarray, col: str) -> dict:
    """
    Compute a full set of descriptive statistics for a clean numeric array.
    """
    n  = len(arr)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    return {
        "col"     : col,
        "n"       : n,
        "mean"    : float(np.mean(arr)),
        "median"  : float(np.median(arr)),
        "std"     : float(np.std(arr, ddof=1))  if n > 1 else float("nan"),
        "sem"     : float(stats.sem(arr))        if n > 1 else float("nan"),
        "min"     : float(np.min(arr)),
        "max"     : float(np.max(arr)),
        "q1"      : q1,
        "q3"      : q3,
        "iqr"     : q3 - q1,
        "skewness": float(stats.skew(arr))       if n > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(arr))   if n > 3 else float("nan"),
    }


def display_descriptive_stats(d: dict) -> None:
    """Print descriptive statistics with skewness and kurtosis interpretation."""
    print_section("DESCRIPTIVE STATISTICS")
    print(f"\n  Variable : {d['col']}")
    print(f"  N        : {d['n']:,}\n")

    rows = [
        ("Sample Size (n)",    fmt_val(d["n"])),
        ("Mean",               fmt_val(d["mean"])),
        ("Median",             fmt_val(d["median"])),
        ("Std Deviation",      fmt_val(d["std"])),
        ("Std Error of Mean",  fmt_val(d["sem"])),
        ("Minimum",            fmt_val(d["min"])),
        ("Maximum",            fmt_val(d["max"])),
        ("25th Percentile",    fmt_val(d["q1"])),
        ("75th Percentile",    fmt_val(d["q3"])),
        ("IQR (Q3 - Q1)",      fmt_val(d["iqr"])),
        ("Skewness",           fmt_val(d["skewness"])),
        ("Kurtosis (excess)",  fmt_val(d["kurtosis"])),
    ]

    label_w = 30
    value_w = 18
    print(f"  {'Statistic':<{label_w}} {'Value':>{value_w}}")
    print(f"  {'-'*label_w} {'-'*value_w}")
    for label, val in rows:
        print(f"  {label:<{label_w}} {val:>{value_w}}")

    sk = d["skewness"]
    ku = d["kurtosis"]

    print(f"\n  Skewness note:")
    if not math.isnan(sk):
        if abs(sk) < 0.5:
            sk_note = "Approximately symmetric distribution."
        elif abs(sk) < 1.0:
            sk_note = f"Moderate {'positive' if sk > 0 else 'negative'} skew."
        else:
            sk_note = (f"Substantial {'positive (right)' if sk > 0 else 'negative (left)'} "
                       f"skew — departure from normality likely.")
        print(f"    Skewness = {sk:,.4f} → {sk_note}")
    else:
        print(f"    Skewness: N/A (insufficient data)")

    print(f"  Kurtosis note:")
    if not math.isnan(ku):
        if abs(ku) < 1.0:
            ku_note = "Mesokurtic — normal-like tail weight."
        elif ku > 1.0:
            ku_note = "Leptokurtic — heavier tails than normal; outliers likely."
        else:
            ku_note = "Platykurtic — lighter tails than normal."
        print(f"    Excess Kurtosis = {ku:,.4f} → {ku_note}")
    else:
        print(f"    Kurtosis: N/A (insufficient data)")


# ======================================================================
# ANDERSON-DARLING TEST
# ======================================================================

def run_anderson_test(arr: np.ndarray) -> dict:
    """
    Run the Anderson-Darling normality test.

    Uses two scipy API paths:
        1. anderson(arr, dist='norm')            — critical value table
        2. anderson(arr, dist='norm',
                    method='interpolate')        — bounded p-value (scipy >= 1.17)

    The critical value table is the PRIMARY decision mechanism.
    The interpolated p-value is a SECONDARY convenience metric, bounded
    to the range [0.01, 0.15] by scipy's lookup table. Values outside
    this range are reported as ≤ 0.01 or ≥ 0.15.

    Returns:
        dict with keys:
            ran              : bool — True if test succeeded
            n                : int
            statistic        : float | None  — A² statistic
            critical_values  : list[float] | None
            significance_levels: list[float] | None — [15,10,5,2.5,1]
            decisions        : list[str] | None  — per-level verdict
            reject_at_5pct   : bool | None — primary decision
            normal_at_5pct   : bool | None — True = fail to reject at 5%
            p_value          : float | None — bounded interpolated p-value
            p_value_note     : str — explains bounded nature
            verdict          : str — human-readable final verdict
            note             : str — populated on failure
    """
    n = len(arr)

    try:
        res = anderson(arr, dist='norm')
    except Exception as exc:
        return {
            "ran": False, "n": n,
            "statistic": None, "critical_values": None,
            "significance_levels": None, "decisions": None,
            "reject_at_5pct": None, "normal_at_5pct": None,
            "p_value": None, "p_value_note": "",
            "verdict": "TEST FAILED",
            "note": f"scipy.stats.anderson() raised an exception: {exc}",
        }

    a2 = float(res.statistic)

    if math.isnan(a2):
        return {
            "ran": False, "n": n,
            "statistic": None, "critical_values": None,
            "significance_levels": None, "decisions": None,
            "reject_at_5pct": None, "normal_at_5pct": None,
            "p_value": None, "p_value_note": "",
            "verdict": "TEST RETURNED NaN",
            "note": (
                "Anderson-Darling returned NaN. "
                "This typically occurs with constant or degenerate data. "
                "Please check your data for constant values or zero variance."
            ),
        }

    critical_values   = [float(cv) for cv in res.critical_values]
    significance_levels = [float(sl) for sl in res.significance_level]

    # Per-level decisions: reject if A² > critical value
    decisions = []
    for sl, cv in zip(significance_levels, critical_values):
        if a2 > cv:
            decisions.append(f"REJECT (evidence against normality at α={sl:.1f}%)")
        else:
            decisions.append(f"Fail to Reject (no evidence against normality at α={sl:.1f}%)")

    # Primary decision at 5% significance level
    # scipy returns significance levels [15,10,5,2.5,1] → index 2 is 5%
    idx_5pct = significance_levels.index(5.0) if 5.0 in significance_levels else 2
    cv_5pct  = critical_values[idx_5pct]
    reject_at_5pct = bool(a2 > cv_5pct)
    normal_at_5pct = not reject_at_5pct

    # Determine the strictest level at which we reject (for nuance)
    strictest_reject = None
    for sl, cv in zip(significance_levels, critical_values):
        if a2 > cv:
            strictest_reject = sl    # last one set is most stringent (1%)

    # Interpolated p-value (scipy >= 1.17)
    p_value      = None
    p_value_note = ""
    try:
        res_p   = anderson(arr, dist='norm', method='interpolate')
        p_value = float(res_p.pvalue)
        # Document bounded nature — scipy table only covers [0.01, 0.15]
        if p_value >= 0.15:
            p_value_note = (
                "p ≥ 0.15 (reported as 0.15): true p-value exceeds the upper bound "
                "of scipy's interpolation table. The actual p-value is ≥ 0.15."
            )
        elif p_value <= 0.01:
            p_value_note = (
                "p ≤ 0.01 (reported as 0.01): true p-value is below the lower bound "
                "of scipy's interpolation table. The actual p-value is ≤ 0.01."
            )
        else:
            p_value_note = (
                "Interpolated from scipy's pre-calculated table. "
                "Bounded to range [0.01, 0.15]."
            )
    except Exception:
        p_value      = None
        p_value_note = "Interpolated p-value not available (scipy < 1.17 or method error)."

    # Verdict
    if reject_at_5pct:
        if strictest_reject == 1.0:
            verdict = "NON-NORMAL — Reject H₀ at α=1% (strong evidence against normality)"
        elif strictest_reject == 2.5:
            verdict = "NON-NORMAL — Reject H₀ at α=2.5%"
        else:
            verdict = "NON-NORMAL — Reject H₀ at α=5%"
    else:
        verdict = "APPROXIMATELY NORMAL — Fail to Reject H₀ at α=5%"

    return {
        "ran"               : True,
        "n"                 : n,
        "statistic"         : a2,
        "critical_values"   : critical_values,
        "significance_levels": significance_levels,
        "decisions"         : decisions,
        "reject_at_5pct"    : reject_at_5pct,
        "normal_at_5pct"    : normal_at_5pct,
        "p_value"           : p_value,
        "p_value_note"      : p_value_note,
        "verdict"           : verdict,
        "note"              : "",
    }


def display_test_results(result: dict) -> None:
    """Display Anderson-Darling test results with the full critical value table."""
    print_section("ANDERSON-DARLING TEST RESULTS")

    n = result["n"]
    print(f"\n  Test             : Anderson-Darling Normality Test")
    print(f"  Distribution     : Normal (Gaussian)")
    print(f"  Sample Size (n)  : {n:,}")
    print(f"  Null Hypothesis  : H₀ — Data follows a normal distribution")
    print(f"  Alt. Hypothesis  : H₁ — Data does NOT follow a normal distribution")
    print(f"  Primary α level  : 0.05  (5%)\n")

    if not result["ran"]:
        print(f"  {SEPARATOR2[:68]}")
        print(f"  Test Status : {result['verdict']}")
        print(f"  {SEPARATOR2[:68]}")
        print(f"\n  {result['note']}")
        return

    a2  = result["statistic"]
    cvs = result["critical_values"]
    sls = result["significance_levels"]

    print(f"  {SEPARATOR2[:68]}")
    print(f"  A² Statistic          : {a2:.6f}")

    if result["p_value"] is not None:
        pv = result["p_value"]
        if pv >= 0.15:
            pv_str = "≥ 0.15"
        elif pv <= 0.01:
            pv_str = "≤ 0.01"
        else:
            pv_str = f"  {pv:.4f}"
        print(f"  Interpolated p-value  : {pv_str}  (bounded: see note below)")
    else:
        print(f"  Interpolated p-value  : N/A")
    print(f"  {SEPARATOR2[:68]}\n")

    # Critical value table
    print(f"  CRITICAL VALUE TABLE")
    print(f"  {'Significance Level':>22}  {'Critical Value':>16}  {'A² Statistic':>14}  {'Decision'}")
    print(f"  {'-'*22}  {'-'*16}  {'-'*14}  {'-'*38}")

    for sl, cv, dec in zip(sls, cvs, result["decisions"]):
        marker = " ←" if sl == 5.0 else ""
        reject = "REJECT" in dec
        sym    = "✘" if reject else "✔"
        label  = f"REJECT H₀" if reject else "Fail to Reject H₀"
        print(f"  {sl:>21.1f}%  {cv:>16.4f}  {a2:>14.4f}  {sym} {label}{marker}")

    print(f"\n  ← Primary decision level (α = 5%)")

    # p-value note
    if result["p_value"] is not None:
        print(f"\n  p-value note: {result['p_value_note']}")

    # Final verdict
    print(f"\n  {SEPARATOR2[:68]}")
    print(f"  FINAL VERDICT : {result['verdict']}")
    print(f"  {SEPARATOR2[:68]}")

    # Large-sample advisory
    if n > 200 and result["reject_at_5pct"]:
        print(f"\n  [LARGE SAMPLE ADVISORY] n = {n:,}.")
        print(f"  Anderson-Darling is highly sensitive at large n. The rejection")
        print(f"  at α=5% may reflect a trivial departure from normality that has")
        print(f"  no practical significance. Examine the A² value relative to the")
        print(f"  critical values and supplement with visual methods (Q-Q plot,")
        print(f"  histogram). Many parametric tests remain valid for n ≥ 30 due")
        print(f"  to the Central Limit Theorem.")

    if n < 20:
        print(f"\n  [SMALL SAMPLE ADVISORY] n = {n:,}.")
        print(f"  Anderson-Darling has low statistical power for small samples.")
        print(f"  Failing to reject H₀ does NOT confirm normality.")
        print(f"  Visual inspection is essential.")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(result: dict, desc: dict, col: str) -> None:
    """Display statistical and clinical interpretation of the AD result."""
    print_section("INTERPRETATION")

    if not result["ran"]:
        print(f"\n  No interpretation available: test was not performed.")
        print(f"  Reason: {result['verdict']}")
        return

    n              = result["n"]
    normal         = bool(result["normal_at_5pct"])   # bool() — safe against numpy.bool_
    a2             = result["statistic"]
    sk             = desc["skewness"]
    ku             = desc["kurtosis"]

    print(f"\n  VARIABLE        : '{col}'")
    print(f"  SAMPLE SIZE     : {n:,}")
    print(f"  A² STATISTIC    : {a2:.6f}")
    print(f"  CONCLUSION      : {result['verdict']}")

    print(f"\n  WHAT THIS MEANS:")
    if normal:
        print(f"  The Anderson-Darling Test does NOT provide evidence against normality")
        print(f"  at the 5% significance level.")
        print(f"  A² = {a2:.6f} does not exceed the critical value at α=5%.")
        print(f"  '{col}' appears to be approximately normally distributed.")
        print(f"  Parametric statistical tests are appropriate.")
    else:
        print(f"  The Anderson-Darling Test provides evidence against normality.")
        print(f"  A² = {a2:.6f} exceeds the critical value at α=5%.")
        print(f"  '{col}' does NOT appear to follow a normal distribution.")
        print(f"  Parametric tests that assume normality may be inappropriate.")

    # Corroborating evidence from shape statistics
    if not (math.isnan(sk) or math.isnan(ku)):
        print(f"\n  CORROBORATING DESCRIPTIVE EVIDENCE:")
        print(f"    Skewness        : {sk:,.4f}", end="")
        if abs(sk) < 0.5:
            print(f"  (approximately symmetric — consistent with normality)")
        elif abs(sk) < 1.0:
            print(f"  (moderate {'positive' if sk > 0 else 'negative'} skew)")
        else:
            print(f"  (substantial skew — supports departure from normality)")

        print(f"    Excess Kurtosis : {ku:,.4f}", end="")
        if abs(ku) < 1.0:
            print(f"  (mesokurtic — normal-like tail weight)")
        elif ku > 1.0:
            print(f"  (leptokurtic — heavier tails, outliers likely)")
        else:
            print(f"  (platykurtic — lighter tails than normal)")

    # Consistency advisory between AD result and shape statistics
    if normal and not math.isnan(sk) and abs(sk) >= 1.0:
        print(f"\n  [ADVISORY] Fail-to-reject at α=5%, but skewness = {sk:.4f} indicates")
        print(f"             meaningful asymmetry. For small n, the test may lack power.")
        print(f"             Visual inspection (Q-Q plot, histogram) is recommended.")

    if not normal and not math.isnan(sk) and abs(sk) < 0.3:
        print(f"\n  [ADVISORY] Rejection at α=5%, but skewness = {sk:.4f} is near zero.")
        print(f"             The departure may be driven by tail shape (kurtosis)")
        print(f"             rather than skewness. A Q-Q plot will clarify this.")

    # Anderson-Darling vs Shapiro-Wilk comparison note
    print(f"\n  ANDERSON-DARLING VS SHAPIRO-WILK:")
    print(f"  • Anderson-Darling places more weight on the TAILS of the distribution.")
    print(f"    It is more sensitive to tail departures than Shapiro-Wilk.")
    print(f"  • Shapiro-Wilk is generally more powerful for detecting non-normality")
    print(f"    in the body of the distribution for small-to-medium samples.")
    print(f"  • Both tests should agree for clear-cut cases. Disagreements suggest")
    print(f"    a borderline distribution where visual inspection is essential.")
    print(f"  • Neither test is definitive — always combine with visual methods.")


def display_medical_recommendations(result: dict, col: str) -> None:
    """Display recommended statistical tests based on the normality result."""
    print_section("MEDICAL RECOMMENDATIONS")

    if not result["ran"]:
        print(f"\n  No recommendations available: Anderson-Darling was not performed.")
        return

    normal = bool(result["normal_at_5pct"])

    if normal:
        print(f"\n  '{col}' appears APPROXIMATELY NORMALLY DISTRIBUTED.")
        print(f"  The following PARAMETRIC tests are appropriate:\n")
        print(f"  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  Test                  │ Use Case                               │")
        print(f"  ├─────────────────────────────────────────────────────────────────┤")
        print(f"  │  Independent T-Test    │ Compare 2 independent group means      │")
        print(f"  │  Paired T-Test         │ Compare pre vs post (same subjects)     │")
        print(f"  │  One-Way ANOVA         │ Compare 3+ independent group means     │")
        print(f"  │  Pearson Correlation   │ Linear association between variables   │")
        print(f"  │  Linear Regression     │ Predict outcome from continuous var    │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")
        print(f"\n  Module paths in VM-WIT-STATS:")
        print(f"    hypothesis_tests/t_test.py")
        print(f"    hypothesis_tests/anova_test.py")
        print(f"    correlation/pearson.py")
        print(f"    regression/linear_regression.py")
    else:
        print(f"\n  '{col}' does NOT appear to be normally distributed.")
        print(f"  The following NON-PARAMETRIC tests are recommended:\n")
        print(f"  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  Test                       │ Use Case                          │")
        print(f"  ├─────────────────────────────────────────────────────────────────┤")
        print(f"  │  Mann-Whitney U Test         │ Compare 2 independent groups     │")
        print(f"  │  Wilcoxon Signed-Rank Test   │ Compare paired measurements      │")
        print(f"  │  Kruskal-Wallis Test         │ Compare 3+ independent groups    │")
        print(f"  │  Friedman Test               │ Repeated measures (3+ timepoints)│")
        print(f"  │  Spearman Correlation        │ Monotonic association             │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")
        print(f"\n  Module paths in VM-WIT-STATS:")
        print(f"    hypothesis_tests/mann_whitney_u_test.py")
        print(f"    hypothesis_tests/wilcoxon_signed_rank_test.py")
        print(f"    hypothesis_tests/kruskal_wallis_test.py")
        print(f"    hypothesis_tests/friedman_test.py")
        print(f"    correlation/spearman.py")

    print(f"\n  IMPORTANT CAVEATS:")
    print(f"  • A² not exceeding the critical value does NOT prove normality.")
    print(f"  • Many parametric tests remain valid for n ≥ 30 (Central Limit Theorem).")
    print(f"  • For n > 200, significant results may reflect trivial tail differences.")
    print(f"  • Always supplement with visual inspection: histogram, Q-Q plot, boxplot.")
    print(f"  • Consider running the Shapiro-Wilk test for comparison.")


def display_limitations(result: dict) -> None:
    """Display limitations and caveats specific to the Anderson-Darling test."""
    print_section("LIMITATIONS")

    n = result["n"]

    print(f"\n  1. NO DEFINITIVE P-VALUE")
    print(f"     Unlike most statistical tests, Anderson-Darling does not provide")
    print(f"     a single, exact p-value. The decision is based on comparing A²")
    print(f"     against tabulated critical values at fixed significance levels.")
    print(f"     The interpolated p-value provided by scipy (method='interpolate')")
    print(f"     is bounded to [0.01, 0.15] by the lookup table.")

    print(f"\n  2. TAIL SENSITIVITY")
    print(f"     Anderson-Darling weights the tails of the distribution more heavily")
    print(f"     than the Kolmogorov-Smirnov test. It may flag data as non-normal")
    print(f"     due to outliers or heavy tails, even when the body of the distribution")
    print(f"     is approximately normal.")

    print(f"\n  3. SAMPLE SIZE SENSITIVITY")
    if n < 20:
        print(f"     Your sample (n = {n:,}) is SMALL. Low power — non-rejection")
        print(f"     does not confirm normality.")
    elif n > 200:
        print(f"     Your sample (n = {n:,}) is LARGE. High power — even trivial")
        print(f"     deviations may produce rejection. The practical significance")
        print(f"     of rejection should be assessed with visual tools.")
    else:
        print(f"     Sample size (n = {n:,}) is in the adequate range for this test.")

    print(f"\n  4. NORMALITY NOT REQUIRED FOR MANY TESTS")
    print(f"     The Central Limit Theorem ensures that for n ≥ 30, the sampling")
    print(f"     distribution of the mean is approximately normal, making many")
    print(f"     parametric tests robust to moderate non-normality.")

    print(f"\n  5. NO ADJUSTMENT FOR MULTIPLE TESTING")
    print(f"     If testing multiple columns, the probability of at least one false")
    print(f"     positive increases. Apply Bonferroni correction or FDR control.")

    print(f"\n  6. ALTERNATIVES")
    print(f"     • Shapiro-Wilk   : Better power for body of distribution (n ≤ 5000).")
    print(f"     • KS Test        : Valid for any distribution but less sensitive.")
    print(f"     • Q-Q Plot       : Visual — always recommended alongside formal tests.")
    print(f"     • Histogram      : Simple visual assessment.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(result: dict,
                 desc: dict,
                 col: str,
                 dataset_path: str) -> str:
    """
    Assemble the complete text report as a single string.

    All boolean checks use bool() coercion to prevent numpy.bool_ issues.
    The report never claims to display a traditional p-value as the primary
    decision metric — only the critical value table comparison is primary.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n  = result["n"]

    lines = []

    # ── Header ──────────────────────────────────────────────────────
    lines += [
        SEPARATOR,
        "  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "  Module  : Anderson-Darling Normality Test",
        f"  Dataset : {dataset_path}",
        f"  Column  : {col}",
        f"  Date    : {ts}",
        SEPARATOR,
    ]

    # ── Descriptive Statistics ───────────────────────────────────────
    lines += [
        "",
        "  DESCRIPTIVE STATISTICS",
        SEPARATOR2,
        f"  Variable              : {desc['col']}",
        f"  Sample Size (n)       : {desc['n']:,}",
        f"  Mean                  : {fmt_val(desc['mean'])}",
        f"  Median                : {fmt_val(desc['median'])}",
        f"  Std Deviation         : {fmt_val(desc['std'])}",
        f"  Std Error of Mean     : {fmt_val(desc['sem'])}",
        f"  Minimum               : {fmt_val(desc['min'])}",
        f"  Maximum               : {fmt_val(desc['max'])}",
        f"  25th Percentile (Q1)  : {fmt_val(desc['q1'])}",
        f"  75th Percentile (Q3)  : {fmt_val(desc['q3'])}",
        f"  IQR (Q3 - Q1)         : {fmt_val(desc['iqr'])}",
        f"  Skewness              : {fmt_val(desc['skewness'])}",
        f"  Kurtosis (excess)     : {fmt_val(desc['kurtosis'])}",
    ]

    # ── Test Results ─────────────────────────────────────────────────
    lines += [
        "",
        "  ANDERSON-DARLING TEST RESULTS",
        SEPARATOR2,
        "  H₀: Data follows a normal distribution.",
        "  H₁: Data does NOT follow a normal distribution.",
        "  Primary significance level (α): 0.05",
        "",
    ]

    if not result["ran"]:
        lines += [
            f"  Test Status : {result['verdict']}",
            "",
            f"  {result['note']}",
        ]
    else:
        a2  = result["statistic"]
        cvs = result["critical_values"]
        sls = result["significance_levels"]

        lines.append(f"  A² Statistic            : {a2:.6f}")

        if result["p_value"] is not None:
            pv = result["p_value"]
            if pv >= 0.15:
                pv_str = "≥ 0.15"
            elif pv <= 0.01:
                pv_str = "≤ 0.01"
            else:
                pv_str = f"{pv:.4f}"
            lines.append(f"  Interpolated p-value    : {pv_str}  (bounded to [0.01, 0.15])")
            lines.append(f"  p-value note            : {result['p_value_note']}")
        else:
            lines.append(f"  Interpolated p-value    : N/A")

        lines += [
            "",
            "  CRITICAL VALUE TABLE",
            f"  {'Significance Level':>22}  {'Critical Value':>16}  {'A²':>10}  Decision",
            f"  {'-'*22}  {'-'*16}  {'-'*10}  {'-'*32}",
        ]

        for sl, cv, dec in zip(sls, cvs, result["decisions"]):
            reject = "REJECT" in dec
            label  = "REJECT H₀" if reject else "Fail to Reject H₀"
            primary = " [PRIMARY]" if sl == 5.0 else ""
            lines.append(
                f"  {sl:>21.1f}%  {cv:>16.4f}  {a2:>10.4f}  {label}{primary}"
            )

        lines += [
            "",
            f"  FINAL VERDICT : {result['verdict']}",
        ]

        if n > 200 and result["reject_at_5pct"]:
            lines += [
                "",
                f"  [LARGE SAMPLE NOTE] n = {n:,}. Rejection at α=5% may reflect a trivial",
                "  departure from normality. Visual inspection is strongly recommended.",
            ]

    # ── Interpretation ───────────────────────────────────────────────
    lines += ["", "  INTERPRETATION", SEPARATOR2]

    if result["ran"]:
        normal = bool(result["normal_at_5pct"])
        a2     = result["statistic"]
        if normal:
            lines += [
                f"  Anderson-Darling does NOT provide evidence against normality at α=5%.",
                f"  A² = {a2:.6f} does not exceed the critical value at the 5% level.",
                f"  '{col}' appears approximately normally distributed.",
                f"  Parametric statistical tests are appropriate.",
            ]
        else:
            lines += [
                f"  Anderson-Darling provides evidence against normality at α=5%.",
                f"  A² = {a2:.6f} exceeds the critical value at the 5% level.",
                f"  '{col}' does not appear to follow a normal distribution.",
                f"  Non-parametric alternatives should be considered.",
            ]
    else:
        lines += [
            f"  Test was not performed: {result['verdict']}",
            "  No interpretation available.",
        ]

    # ── Medical Recommendations ──────────────────────────────────────
    lines += ["", "  MEDICAL RECOMMENDATIONS", SEPARATOR2]

    if result["ran"] and bool(result["normal_at_5pct"]):
        lines += [
            "  Recommended PARAMETRIC tests:",
            "    • Independent T-Test     — hypothesis_tests/t_test.py",
            "    • Paired T-Test          — hypothesis_tests/t_test.py",
            "    • One-Way ANOVA          — hypothesis_tests/anova_test.py",
            "    • Pearson Correlation    — correlation/pearson.py",
            "    • Linear Regression      — regression/linear_regression.py",
        ]
    elif result["ran"]:
        lines += [
            "  Recommended NON-PARAMETRIC tests:",
            "    • Mann-Whitney U Test        — hypothesis_tests/mann_whitney_u_test.py",
            "    • Wilcoxon Signed-Rank Test  — hypothesis_tests/wilcoxon_signed_rank_test.py",
            "    • Kruskal-Wallis Test        — hypothesis_tests/kruskal_wallis_test.py",
            "    • Friedman Test              — hypothesis_tests/friedman_test.py",
            "    • Spearman Correlation       — correlation/spearman.py",
        ]
    else:
        lines.append("  Run normality test successfully to receive recommendations.")

    # ── Limitations ──────────────────────────────────────────────────
    lines += [
        "",
        "  LIMITATIONS",
        SEPARATOR2,
        "  • Anderson-Darling decision is based on critical value comparison, NOT p-value.",
        "  • The interpolated p-value is bounded to [0.01, 0.15] by scipy's table.",
        "  • Anderson-Darling weights tails heavily — sensitive to outliers.",
        "  • Small samples (n < 20): low power; non-rejection ≠ normality.",
        "  • Large samples (n > 200): may reject for trivially small deviations.",
        "  • Parametric tests are often robust for n ≥ 30 (Central Limit Theorem).",
        "  • Always supplement with visual methods: histogram, Q-Q plot.",
        "  • Consider also: Shapiro-Wilk (shapiro_test.py), KS Test (ks_test.py).",
        "  • Multiple testing: apply Bonferroni correction when testing many columns.",
    ]

    # ── Footer ───────────────────────────────────────────────────────
    lines += [
        "",
        SEPARATOR,
        "  END OF REPORT",
        SEPARATOR,
        "",
    ]

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """Write the report to outputs/anderson_test_result.txt."""
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

def run_anderson_test_module() -> None:
    """
    Interactive entry point for the Anderson-Darling Normality Test module.

    Steps:
        1.  Load CSV
        2.  Display columns
        3.  Select column
        4.  Clean data
        5.  Descriptive statistics
        6.  Run Anderson-Darling test
        7.  Display results
        8.  Interpretation
        9.  Medical recommendations
        10. Limitations
        11. Build and save report
    """
    print_header()
    print("  This module performs the Anderson-Darling Normality Test.")
    print("  It evaluates whether a continuous variable follows a normal distribution.")
    print("  Unlike Shapiro-Wilk, there is no hard upper limit on sample size.")
    print("  Decision is based on critical value comparison, not a single p-value.")

    # Step 1
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2
    display_columns(df)

    # Step 3
    col = select_numeric_column(df)
    if col is None:
        sys.exit(1)

    # Step 4
    arr = clean_column(df, col)
    if arr is None:
        sys.exit(1)

    # Step 5
    desc = compute_descriptive_stats(arr, col)
    display_descriptive_stats(desc)

    # Step 6 & 7
    print_section("RUNNING ANDERSON-DARLING TEST")
    result = run_anderson_test(arr)
    display_test_results(result)

    # Step 8
    display_interpretation(result, desc, col)

    # Step 9
    display_medical_recommendations(result, col)

    # Step 10
    display_limitations(result)

    # Step 11
    report_text = build_report(result, desc, col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Anderson-Darling Normality Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_anderson_test_module()