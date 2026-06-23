# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Shapiro-Wilk Normality Test
# File        : normality_tests/shapiro_test.py
# Description : Tests whether a continuous variable follows a normal
#               distribution. Uses scipy.stats.shapiro().
#               Appropriate for sample sizes 3 ≤ n ≤ 5000.
#               For n > 5000, skips the test and advises alternatives.
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "shapiro_test_result.txt")

# Shapiro-Wilk is unreliable above this threshold
SHAPIRO_MAX_N = 5000

# Minimum meaningful sample size for Shapiro-Wilk
SHAPIRO_MIN_N = 3

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Shapiro-Wilk Normality Test
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
    """Format a value for display, handling None, NaN, Inf gracefully."""
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

    Uses range (max - min) < tol rather than std, because std can be
    non-zero due to floating-point rounding even for constant arrays.
    This must be checked BEFORE calling shapiro(), which returns
    W=1.0, p=1.0 for constant data — a misleading 'Normal' result.
    """
    if len(arr) == 0:
        return True
    return bool(float(np.max(arr)) - float(np.min(arr)) < tol)


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_shapiro.csv",
                         n: int = 100,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic clinical dataset for Shapiro-Wilk testing.

    Columns:
        Patient_ID  : unique subject identifier
        systolic_bp : continuous, approximately normally distributed
        hba1c       : continuous, right-skewed (non-normal)
        age         : integer, approximately normal
        creatinine  : continuous, right-skewed

    Usage:
        from normality_tests.shapiro_test import generate_sample_csv
        generate_sample_csv("data/sample_shapiro.csv")
    """
    rng = np.random.default_rng(seed)

    systolic_bp = rng.normal(loc=125.0, scale=15.0, size=n).clip(80, 200).round(1)
    hba1c       = rng.exponential(scale=1.5, size=n) + 5.0               # right-skewed
    hba1c       = hba1c.clip(4.0, 15.0).round(1)
    age         = rng.normal(loc=55.0, scale=12.0, size=n).clip(18, 90).round(0).astype(int)
    creatinine  = rng.lognormal(mean=0.1, sigma=0.4, size=n).clip(0.5, 10.0).round(2)

    df = pd.DataFrame({
        "Patient_ID" : [f"PT{str(i + 1).zfill(4)}" for i in range(n)],
        "systolic_bp": systolic_bp,
        "hba1c"      : hba1c,
        "age"        : age,
        "creatinine" : creatinine,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows     : {n}")
    print(f"       Columns  : Patient_ID, systolic_bp, hba1c, age, creatinine")
    print(f"       Notes    : systolic_bp and age are approximately normal;")
    print(f"                  hba1c and creatinine are right-skewed (non-normal).")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    """Prompt user for CSV file path with validation loop."""
    print_section("STEP 1 | LOAD DATASET")
    print("\n  NOTE: The Shapiro-Wilk Test evaluates a single continuous variable.")
    print("        Each row should represent one observation (patient/subject).")
    print("        Example: Patient_ID | systolic_bp | hba1c | creatinine")
    print("\n  TIP: To generate a sample dataset for testing, run:")
    print("       from normality_tests.shapiro_test import generate_sample_csv")
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

    Validates:
        - Column exists in the dataframe.
        - Column contains at least some numeric values.
    Non-numeric values in an otherwise numeric column are flagged as a warning.
    The hard numeric check (n >= 3, not constant) happens after cleaning.
    """
    print_section("STEP 3 | SELECT COLUMN FOR NORMALITY TEST")
    print("\n  Select ONE continuous numeric column to test for normality.")
    print("  Examples from SUPPORT2: 'charges', 'meanbp', 'age', 'hrt', 'crea'")
    print("  Examples from NHANES  : 'systolic_bp', 'hba1c', 'BMI', 'cholesterol'")

    while True:
        col = input("\n  Enter column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

        # Try numeric coercion to check usability
        coerced    = pd.to_numeric(df[col], errors="coerce")
        valid_n    = int(coerced.notna().sum())
        total_n    = len(coerced)
        non_num    = total_n - valid_n

        if valid_n == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values.")
            print(f"          Please select a continuous numeric column.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [WARNING] Column '{col}' has dtype '{df[col].dtype}'.")
            print(f"            {valid_n:,} of {total_n:,} values are numeric.")
            print(f"            {non_num:,} non-numeric value(s) will be treated as missing.")
        elif non_num > 0:
            print(f"  [WARNING] {non_num:,} non-numeric/missing value(s) in '{col}'.")
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
        2. Drop NaN values.
        3. Drop ±Inf values.
        4. Report counts at each step.
        5. Hard stop if n < SHAPIRO_MIN_N (3).
        6. Hard stop if column is effectively constant.

    Returns:
        Clean numpy float64 array, or None if analysis cannot proceed.
    """
    print_section("STEP 4 | DATA CLEANING & VALIDATION")

    series = pd.to_numeric(df[col], errors="coerce")
    original_n = len(series)
    print(f"\n  Column          : '{col}'")
    print(f"  Original rows   : {original_n:,}")

    # Drop NaN
    nan_count = int(series.isna().sum())
    series    = series.dropna()
    if nan_count > 0:
        print(f"  Dropped (NaN)   : {nan_count:,} row(s)")

    # Drop ±Inf
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

    # Hard stop: insufficient observations
    if clean_n < SHAPIRO_MIN_N:
        print(f"\n  [ERROR] Only {clean_n:,} usable value(s) remain after cleaning.")
        print(f"          Shapiro-Wilk Test requires at least {SHAPIRO_MIN_N} observations.")
        print(f"          Cannot proceed with normality testing.")
        return None

    # Hard stop: constant column
    if is_effectively_constant(arr):
        print(f"\n  [ERROR] Column '{col}' is effectively constant.")
        print(f"          All {clean_n:,} values are identical (or differ by < 1e-10).")
        print(f"          Normality is undefined for a constant variable.")
        print(f"          Note: scipy.stats.shapiro() returns W=1.0, p=1.0 for constant")
        print(f"          data, which would falsely indicate 'Normal'. This is detected")
        print(f"          and blocked to prevent misleading output.")
        return None

    # Small-sample advisory (not a hard stop)
    if clean_n < 20:
        print(f"\n  [WARNING] Sample size is small (n = {clean_n:,}).")
        print(f"            Shapiro-Wilk Test has low power for small samples.")
        print(f"            A non-significant result (p > 0.05) does not confirm normality.")
        print(f"            Interpret results with caution.")
    else:
        print(f"\n  [OK] Data cleaned and validated. Ready for Shapiro-Wilk Test.")

    return arr


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(arr: np.ndarray, col: str) -> dict:
    """
    Compute a full set of descriptive statistics for a clean numeric array.

    Returns a dict with keys:
        col, n, mean, median, std, sem, min, max, q1, q3, iqr,
        skewness, kurtosis
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
    """Print descriptive statistics in a formatted table."""
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

    # Skewness and kurtosis interpretation
    sk = d["skewness"]
    ku = d["kurtosis"]
    print(f"\n  Skewness note  :")
    if not math.isnan(sk):
        if abs(sk) < 0.5:
            sk_note = "Approximately symmetric distribution."
        elif abs(sk) < 1.0:
            sk_note = ("Moderate " + ("positive" if sk > 0 else "negative") +
                       " skew. Right tail heavier." if sk > 0 else
                       "Moderate negative skew. Left tail heavier.")
        else:
            sk_note = ("Substantial " + ("positive (right) skew. " if sk > 0
                       else "negative (left) skew. ") +
                       "Departure from normality likely.")
        print(f"    Skewness = {sk:,.4f} → {sk_note}")
    else:
        print(f"    Skewness: N/A (insufficient data)")

    print(f"  Kurtosis note  :")
    if not math.isnan(ku):
        if abs(ku) < 1.0:
            ku_note = "Close to mesokurtic (normal-like tail weight)."
        elif ku > 1.0:
            ku_note = "Leptokurtic: heavier tails than normal. Outliers likely."
        else:
            ku_note = "Platykurtic: lighter tails than normal."
        print(f"    Excess Kurtosis = {ku:,.4f} → {ku_note}")
    else:
        print(f"    Kurtosis: N/A (insufficient data)")


# ======================================================================
# SHAPIRO-WILK TEST
# ======================================================================

def run_shapiro_test(arr: np.ndarray) -> dict:
    """
    Run the Shapiro-Wilk normality test or apply the large-sample rule.

    Rules:
        n < SHAPIRO_MIN_N (3) : blocked in clean_column(), never reached here.
        n > SHAPIRO_MAX_N (5000): test is NOT run. No fake W or p is produced.
                                  Guidance on alternatives is provided instead.
        3 ≤ n ≤ 5000          : scipy.stats.shapiro() is called.

    Returns a dict with keys:
        ran       : bool  — True if test was actually run
        n         : int
        w_stat    : float | None
        p_value   : float | None
        normal    : bool | None  — True=normal, False=non-normal, None=not assessed
        verdict   : str
        note      : str          — populated when test was skipped
    """
    n = len(arr)

    # Large-sample rule: Shapiro-Wilk is unreliable above 5000
    if n > SHAPIRO_MAX_N:
        return {
            "ran"    : False,
            "n"      : n,
            "w_stat" : None,
            "p_value": None,
            "normal" : None,
            "verdict": "NOT ASSESSED (n > 5000)",
            "note"   : (
                f"Shapiro-Wilk Test was not performed because n = {n:,} exceeds "
                f"the recommended maximum of {SHAPIRO_MAX_N:,}. "
                "For large samples, trivially small deviations from normality "
                "will produce significant p-values that have no practical meaning. "
                "Please use Anderson-Darling or Kolmogorov-Smirnov tests, "
                "and supplement with Q-Q plots and histograms."
            ),
        }

    # Run Shapiro-Wilk
    try:
        raw_stat, raw_p = stats.shapiro(arr)
    except Exception as exc:
        return {
            "ran"    : False,
            "n"      : n,
            "w_stat" : None,
            "p_value": None,
            "normal" : None,
            "verdict": "TEST FAILED",
            "note"   : f"scipy.stats.shapiro() raised an exception: {exc}",
        }

    # Coerce to Python native types immediately to avoid numpy.bool_ issues
    w_stat  = float(raw_stat)
    p_value = float(raw_p)

    # Guard against NaN output (can occur with degenerate input)
    if math.isnan(w_stat) or math.isnan(p_value):
        return {
            "ran"    : False,
            "n"      : n,
            "w_stat" : None,
            "p_value": None,
            "normal" : None,
            "verdict": "TEST RETURNED NaN",
            "note"   : (
                "Shapiro-Wilk returned NaN results. "
                "This may occur with near-constant data or degenerate distributions. "
                "Please check your data."
            ),
        }

    # Determine normality: bool() ensures Python bool, never numpy.bool_
    normal  = bool(p_value > 0.05)
    verdict = "NORMAL (Fail to reject H₀)" if normal else "NON-NORMAL (Reject H₀)"

    return {
        "ran"    : True,
        "n"      : n,
        "w_stat" : w_stat,
        "p_value": p_value,
        "normal" : normal,
        "verdict": verdict,
        "note"   : "",
    }


def display_test_results(result: dict) -> None:
    """Display Shapiro-Wilk test results in a formatted block."""
    print_section("SHAPIRO-WILK TEST RESULTS")

    n = result["n"]

    print(f"\n  Test             : Shapiro-Wilk Normality Test")
    print(f"  Sample Size (n)  : {n:,}")
    print(f"  Null Hypothesis  : H₀ — Data is normally distributed")
    print(f"  Alt. Hypothesis  : H₁ — Data is NOT normally distributed")
    print(f"  Significance (α) : 0.05\n")

    if not result["ran"]:
        print(f"  {SEPARATOR2[:68]}")
        print(f"  Test Status : {result['verdict']}")
        print(f"  {SEPARATOR2[:68]}")
        print(f"\n  {result['note']}")

        if n > SHAPIRO_MAX_N:
            print(f"\n  ALTERNATIVES FOR LARGE SAMPLES (n > {SHAPIRO_MAX_N:,}):")
            print(f"    1. Anderson-Darling Test   → normality_tests/anderson_test.py")
            print(f"    2. Kolmogorov-Smirnov Test → normality_tests/ks_test.py")
            print(f"    3. Q-Q Plot (visual)       → visualization/qq_plot.py")
            print(f"    4. Histogram with normal overlay")
            print(f"\n  NOTE: With very large samples (n > 1000), the Central Limit Theorem")
            print(f"  generally ensures that test statistics (means, sums) are approximately")
            print(f"  normally distributed regardless of the raw data distribution.")
            print(f"  Parametric tests may still be valid. Consult a statistician.")
        return

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'W Statistic':<40} {result['w_stat']:>20.6f}")
    print(f"  {'p-value':<40} {result['p_value']:>20.6f}")
    print(f"  {'Sample Size (n)':<40} {n:>20,}")
    print(f"  {SEPARATOR2[:68]}")

    verdict = result["verdict"]
    normal  = result["normal"]
    print(f"\n  Result  : {verdict}")
    print(f"  p-value : {result['p_value']:.6f}  (α = 0.05)")

    # W statistic closeness to 1.0
    w = result["w_stat"]
    dev = 1.0 - w
    print(f"\n  W Statistic Note:")
    print(f"    W ranges from 0 to 1. A value of 1.0 indicates perfect normality.")
    print(f"    W = {w:.6f}  (deviation from 1.0 = {dev:.6f})")
    if dev < 0.01:
        print(f"    W is very close to 1.0 → strong evidence of normality.")
    elif dev < 0.05:
        print(f"    W is close to 1.0 → mild deviation from normality.")
    elif dev < 0.15:
        print(f"    W shows moderate departure from 1.0 → notable non-normality.")
    else:
        print(f"    W shows substantial departure from 1.0 → strong non-normality.")

    if n < 20 and normal:
        print(f"\n  [CAUTION] n = {n:,} is small. A non-significant p-value (p > 0.05)")
        print(f"             does NOT confirm normality — the test has low power.")
        print(f"             Visual inspection (Q-Q plot, histogram) is recommended.")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(result: dict, desc: dict, col: str) -> None:
    """
    Display statistical and medical interpretation of the Shapiro-Wilk result.
    """
    print_section("INTERPRETATION")

    n      = result["n"]
    normal = result["normal"]

    if not result["ran"]:
        print(f"\n  No interpretation available: test was not performed.")
        print(f"  Reason: {result['verdict']}")
        return

    sk = desc["skewness"]
    ku = desc["kurtosis"]

    print(f"\n  VARIABLE        : '{col}'")
    print(f"  SAMPLE SIZE     : {n:,}")
    print(f"  W STATISTIC     : {result['w_stat']:.6f}")
    print(f"  P-VALUE         : {result['p_value']:.6f}")
    print(f"  CONCLUSION      : {result['verdict']}")

    print(f"\n  WHAT THIS MEANS:")
    if normal:
        print(f"  The Shapiro-Wilk Test does NOT provide evidence against normality.")
        print(f"  (p = {result['p_value']:.4f} > 0.05 → Fail to reject H₀)")
        print(f"  The distribution of '{col}' is approximately normal.")
        print(f"  Parametric statistical tests are appropriate.")
    else:
        print(f"  The Shapiro-Wilk Test provides evidence that '{col}' is NOT normally")
        print(f"  distributed.")
        print(f"  (p = {result['p_value']:.4f} ≤ 0.05 → Reject H₀)")
        print(f"  Parametric tests that assume normality may be inappropriate.")

    # Corroborating evidence from skewness / kurtosis
    if not (math.isnan(sk) or math.isnan(ku)):
        print(f"\n  CORROBORATING DESCRIPTIVE EVIDENCE:")
        print(f"    Skewness        : {sk:,.4f}", end="")
        if abs(sk) < 0.5:
            print(f"  (approximately symmetric)")
        elif abs(sk) < 1.0:
            print(f"  (moderate {'positive' if sk > 0 else 'negative'} skew)")
        else:
            print(f"  (substantial {'positive (right)' if sk > 0 else 'negative (left)'} skew)")

        print(f"    Excess Kurtosis : {ku:,.4f}", end="")
        if abs(ku) < 1.0:
            print(f"  (mesokurtic — normal-like tail weight)")
        elif ku > 1.0:
            print(f"  (leptokurtic — heavier tails, outliers likely)")
        else:
            print(f"  (platykurtic — lighter tails than normal)")

    # Consistency check between p-value and skewness
    if normal and not math.isnan(sk) and abs(sk) >= 1.0:
        print(f"\n  [ADVISORY] p > 0.05 suggests normality, but skewness = {sk:.4f}")
        print(f"             indicates meaningful asymmetry. For small samples,")
        print(f"             Shapiro-Wilk may lack power to detect this departure.")
        print(f"             Visual inspection is strongly recommended.")

    if not normal and not math.isnan(sk) and abs(sk) < 0.5:
        print(f"\n  [ADVISORY] p ≤ 0.05 suggests non-normality, but skewness = {sk:.4f}")
        print(f"             is near-symmetric. The departure from normality may be")
        print(f"             driven by tail behaviour (kurtosis) rather than skew.")
        print(f"             Examine a Q-Q plot for clarification.")


def display_medical_recommendations(result: dict, col: str) -> None:
    """
    Display recommended statistical tests based on the normality result.
    """
    print_section("MEDICAL RECOMMENDATIONS")

    if not result["ran"]:
        print(f"\n  No recommendations available: Shapiro-Wilk was not performed.")
        return

    normal = result["normal"]

    if normal:
        print(f"\n  '{col}' appears to be APPROXIMATELY NORMALLY DISTRIBUTED.")
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
        print(f"  │  Friedman Test               │ Repeated measures (3+ time pts)  │")
        print(f"  │  Spearman Correlation        │ Monotonic association             │")
        print(f"  │  Logistic Regression         │ Binary outcome prediction         │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")
        print(f"\n  Module paths in VM-WIT-STATS:")
        print(f"    hypothesis_tests/mann_whitney_u_test.py")
        print(f"    hypothesis_tests/wilcoxon_signed_rank_test.py")
        print(f"    hypothesis_tests/kruskal_wallis_test.py")
        print(f"    hypothesis_tests/friedman_test.py")
        print(f"    correlation/spearman.py")
        print(f"    regression/logistic_regression.py")

    print(f"\n  IMPORTANT CAVEATS:")
    print(f"  • A non-significant Shapiro-Wilk (p > 0.05) does NOT PROVE normality.")
    print(f"    It only means the test lacks evidence to reject it.")
    print(f"  • For small samples (n < 30), Shapiro-Wilk has low statistical power.")
    print(f"  • For large samples (n > 200), even trivial deviations may be significant.")
    print(f"  • Always supplement with visual inspection: histogram, Q-Q plot, boxplot.")
    print(f"  • Many parametric tests are robust to moderate non-normality when n ≥ 30")
    print(f"    due to the Central Limit Theorem.")


def display_limitations(result: dict) -> None:
    """Display limitations and caveats of the Shapiro-Wilk test."""
    print_section("LIMITATIONS")

    n = result["n"]

    print(f"\n  1. SAMPLE SIZE SENSITIVITY")
    print(f"     Shapiro-Wilk is sensitive to sample size in both directions:")
    print(f"     • Small n (< 20)  : Low power → may fail to detect non-normality.")
    print(f"     • Large n (> 200) : High power → may detect trivial deviations that")
    print(f"       are not practically meaningful.")
    if n < 20:
        print(f"     Your sample (n = {n:,}) is SMALL. Results may be unreliable.")
    elif n > 200:
        print(f"     Your sample (n = {n:,}) is LARGE. A significant p-value may")
        print(f"     reflect trivial deviations rather than meaningful non-normality.")

    print(f"\n  2. THIS TEST IS NOT DEFINITIVE")
    print(f"     • p > 0.05 does not prove normality; it only fails to disprove it.")
    print(f"     • p ≤ 0.05 does not mean the data is drastically non-normal.")
    print(f"     • Always supplement with visual methods (Q-Q plot, histogram).")

    print(f"\n  3. APPLICABLE RANGE")
    print(f"     • Designed and validated for 3 ≤ n ≤ 5000.")
    print(f"     • Behaviour outside this range is unreliable.")

    print(f"\n  4. SINGLE VARIABLE")
    print(f"     • This test evaluates ONE variable at a time.")
    print(f"     • In multivariate analyses, multivariate normality is a separate assumption.")

    print(f"\n  5. NO ADJUSTMENT FOR MULTIPLE TESTING")
    print(f"     • If you run Shapiro-Wilk on many variables, false positives accumulate.")
    print(f"     • Apply Bonferroni correction or FDR control when testing multiple columns.")

    print(f"\n  6. ALTERNATIVES CONSIDERED")
    print(f"     • Anderson-Darling   : More sensitive to tail behaviour.")
    print(f"     • Kolmogorov-Smirnov : Distribution-free, but less powerful for normality.")
    print(f"     • Lilliefors Test    : KS variant with estimated parameters.")
    print(f"     • Q-Q Plot           : Visual, not a formal test — always recommended.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(result: dict,
                 desc: dict,
                 col: str,
                 dataset_path: str) -> str:
    """
    Assemble the complete text report as a single string.

    All verdict text uses bool(result["normal"]) to ensure safe
    comparison regardless of numpy scalar type.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n  = result["n"]

    lines = []

    # ── Header ──────────────────────────────────────────────────────
    lines += [
        SEPARATOR,
        "  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "  Module  : Shapiro-Wilk Normality Test",
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

    # ── Shapiro-Wilk Test Results ────────────────────────────────────
    lines += [
        "",
        "  SHAPIRO-WILK TEST RESULTS",
        SEPARATOR2,
        f"  H₀: Data is normally distributed.",
        f"  H₁: Data is NOT normally distributed.",
        f"  Significance level (α): 0.05",
        "",
    ]

    if not result["ran"]:
        lines += [
            f"  Test Status : {result['verdict']}",
            "",
            f"  {result['note']}",
        ]
        if n > SHAPIRO_MAX_N:
            lines += [
                "",
                "  ALTERNATIVES FOR LARGE SAMPLES:",
                "    1. Anderson-Darling Test   (normality_tests/anderson_test.py)",
                "    2. Kolmogorov-Smirnov Test (normality_tests/ks_test.py)",
                "    3. Q-Q Plot (visual method)",
            ]
    else:
        # Safe verdict: bool() ensures never numpy.bool_
        verdict = "Normal" if bool(result["normal"]) else "Non-Normal"
        lines += [
            f"  W Statistic             : {result['w_stat']:.6f}",
            f"  p-value                 : {result['p_value']:.6f}",
            f"  Sample Size (n)         : {n:,}",
            f"  Result                  : {verdict}",
            f"  Verdict                 : {result['verdict']}",
        ]

    # ── Interpretation ───────────────────────────────────────────────
    lines += ["", "  INTERPRETATION", SEPARATOR2]

    if result["ran"]:
        normal = bool(result["normal"])
        if normal:
            lines += [
                f"  The Shapiro-Wilk Test does NOT provide evidence against normality.",
                f"  (p = {result['p_value']:.6f} > 0.05 → Fail to reject H₀)",
                f"  '{col}' appears to be approximately normally distributed.",
                f"  Parametric statistical tests are appropriate.",
            ]
        else:
            lines += [
                f"  The Shapiro-Wilk Test provides evidence that '{col}' is NOT normally",
                f"  distributed.",
                f"  (p = {result['p_value']:.6f} ≤ 0.05 → Reject H₀)",
                f"  Parametric tests that assume normality may be inappropriate.",
            ]
    else:
        lines += [
            f"  Test was not performed: {result['verdict']}",
            f"  No interpretation available.",
        ]

    # ── Medical Recommendations ──────────────────────────────────────
    lines += ["", "  MEDICAL RECOMMENDATIONS", SEPARATOR2]

    if result["ran"]:
        if bool(result["normal"]):
            lines += [
                "  Recommended PARAMETRIC tests:",
                "    • Independent T-Test     — hypothesis_tests/t_test.py",
                "    • Paired T-Test          — hypothesis_tests/t_test.py",
                "    • One-Way ANOVA          — hypothesis_tests/anova_test.py",
                "    • Pearson Correlation    — correlation/pearson.py",
                "    • Linear Regression      — regression/linear_regression.py",
            ]
        else:
            lines += [
                "  Recommended NON-PARAMETRIC tests:",
                "    • Mann-Whitney U Test        — hypothesis_tests/mann_whitney_u_test.py",
                "    • Wilcoxon Signed-Rank Test  — hypothesis_tests/wilcoxon_signed_rank_test.py",
                "    • Kruskal-Wallis Test        — hypothesis_tests/kruskal_wallis_test.py",
                "    • Friedman Test              — hypothesis_tests/friedman_test.py",
                "    • Spearman Correlation       — correlation/spearman.py",
            ]
    else:
        lines.append("  Run normality test first to receive recommendations.")

    # ── Limitations ──────────────────────────────────────────────────
    lines += [
        "",
        "  LIMITATIONS",
        SEPARATOR2,
        "  • p > 0.05 does NOT prove normality; it only fails to reject H₀.",
        "  • p ≤ 0.05 does not mean the data is drastically non-normal.",
        "  • Small samples (n < 20): low power — non-significance ≠ normality.",
        "  • Large samples (n > 200): even trivial deviations may be significant.",
        "  • Shapiro-Wilk is validated for 3 ≤ n ≤ 5000 only.",
        "  • Always supplement with visual methods: histogram, Q-Q plot, boxplot.",
        "  • Parametric tests are often robust for n ≥ 30 (Central Limit Theorem).",
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
    """Write the report to outputs/shapiro_test_result.txt."""
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

def run_shapiro_test_module() -> None:
    """
    Interactive entry point for the Shapiro-Wilk Normality Test module.

    Orchestrates all steps:
        1. Load CSV
        2. Display columns
        3. Select column
        4. Clean data
        5. Descriptive statistics
        6. Run Shapiro-Wilk (or apply large-sample rule)
        7. Display results
        8. Interpretation
        9. Medical recommendations
        10. Limitations
        11. Build and save report
    """
    print_header()
    print("  This module performs the Shapiro-Wilk Normality Test.")
    print("  It evaluates whether a continuous variable follows a normal distribution.")
    print("  Validated for sample sizes 3 ≤ n ≤ 5000.")
    print("  For n > 5000, guidance on alternative tests is provided.")

    # Step 1: Load CSV
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select column
    col = select_numeric_column(df)
    if col is None:
        sys.exit(1)

    # Step 4: Clean data
    arr = clean_column(df, col)
    if arr is None:
        sys.exit(1)

    # Step 5: Descriptive statistics
    desc = compute_descriptive_stats(arr, col)
    display_descriptive_stats(desc)

    # Step 6 & 7: Run test and display results
    print_section("RUNNING SHAPIRO-WILK TEST")
    result = run_shapiro_test(arr)
    display_test_results(result)

    # Step 8: Interpretation
    display_interpretation(result, desc, col)

    # Step 9: Medical recommendations
    display_medical_recommendations(result, col)

    # Step 10: Limitations
    display_limitations(result)

    # Step 11: Build and save report
    report_text = build_report(result, desc, col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Shapiro-Wilk Normality Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_shapiro_test_module()