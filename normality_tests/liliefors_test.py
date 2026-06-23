# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Lilliefors Test (Normality)
# File        : normality_tests/lilliefors_test.py
# Description : Tests whether a continuous numeric variable follows a
#               normal distribution when population mean and standard
#               deviation are unknown and estimated from the sample.
#               Correct alternative to standard KS test for normality.
#               Uses statsmodels.stats.diagnostic.lilliefors().
#
# Design Note : The standard Kolmogorov-Smirnov test (scipy.stats.kstest)
#               requires fully pre-specified distribution parameters.
#               When mean and SD are estimated from the sample — as in all
#               real medical datasets — the standard KS test is anti-
#               conservative: it inflates p-values and under-rejects false
#               normality. The Lilliefors correction (1967) re-derives the
#               critical values for the case of estimated parameters,
#               producing a correctly-sized test. This module uses the
#               Lilliefors test exclusively for that reason.
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

try:
    from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "lilliefors_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Lilliefors Test (Normality)
{SEPARATOR}
"""

LARGE_SAMPLE_THRESHOLD = 100

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


def is_effectively_constant(arr: np.ndarray, tol: float = 1e-10) -> bool:
    return bool((arr.max() - arr.min()) < tol)


def check_statsmodels() -> bool:
    """
    Verify statsmodels is installed. Print actionable error if not.
    """
    if not STATSMODELS_AVAILABLE:
        print("\n  [ERROR] Required package 'statsmodels' is not installed.")
        print("          Please install it before using this module:")
        print("          pip install statsmodels")
        print("\n  Statistical reason: The Lilliefors correction requires")
        print("  critical value tables not available in scipy. The standard")
        print("  KS test (scipy.stats.kstest) is anti-conservative when")
        print("  mean and SD are estimated from the sample and must NOT be")
        print("  used as a substitute for normality testing.")
        return False
    return True


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_lilliefors.csv",
                         n: int = 60,
                         seed: int = 42):
    """
    Generate a synthetic dataset for Lilliefors Test testing.

    Columns:
        Patient_ID : unique subject identifier
        SystolicBP : systolic blood pressure (approximately normal)
        HbA1c      : HbA1c values (mildly right-skewed)
        Pain_Score : pain score (ordinal, non-normal)

    Usage:
        from normality_tests.lilliefors_test import generate_sample_csv
        generate_sample_csv("data/sample_lilliefors.csv")
    """
    rng = np.random.default_rng(seed)

    systolic = rng.normal(loc=128.0, scale=15.0, size=n).clip(90, 200).round(1)
    hba1c    = rng.lognormal(mean=2.0, sigma=0.25, size=n).clip(4.0, 14.0).round(2)
    pain     = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           size=n, p=[0.05, 0.08, 0.12, 0.15, 0.20,
                                      0.15, 0.12, 0.07, 0.04, 0.02])

    df = pd.DataFrame({
        "Patient_ID" : [f"PT{str(i + 1).zfill(3)}" for i in range(n)],
        "SystolicBP" : systolic,
        "HbA1c"      : hba1c,
        "Pain_Score" : pain.astype(float),
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows    : {n}")
    print(f"       Columns : Patient_ID, SystolicBP (normal),")
    print(f"                 HbA1c (right-skewed), Pain_Score (ordinal)")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    print("\n  TIP: To generate a sample dataset for testing, run:")
    print("       from normality_tests.lilliefors_test import generate_sample_csv")
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

def select_numeric_column(df: pd.DataFrame) -> str | None:
    print_section("STEP 3 | SELECT VARIABLE")
    print("\n  Select ONE continuous numeric variable to test for normality.")
    print("  Example: age, weight, SystolicBP, HbA1c, charges, meanbp, etc.")

    while True:
        col = input("\n  Enter column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        valid_count = int(coerced.notna().sum())
        total_count = len(coerced)
        non_numeric = total_count - valid_count

        if valid_count == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values.")
            print("          Please choose a different column.")
            continue

        if non_numeric > 0:
            print(f"  [WARNING] {non_numeric:,} non-numeric value(s) in '{col}'"
                  f" will be treated as missing.")

        print(f"  [OK] Column selected: '{col}'")
        print(f"       Valid numeric values: {valid_count:,} / {total_count:,}")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame, col: str) -> np.ndarray | None:
    print_section("STEP 4 | DATA CLEANING & VALIDATION")

    series = pd.to_numeric(df[col], errors="coerce")
    original_n = len(series)
    print(f"\n  Original observations : {original_n:,}")

    # Drop NaN
    before = series.notna().sum()
    series = series.dropna()
    dropped_nan = original_n - len(series)
    if dropped_nan > 0:
        print(f"  Dropped (NaN values)        : {dropped_nan:,}")

    # Drop infinite
    arr = series.to_numpy(dtype=float)
    before_inf = len(arr)
    arr = arr[np.isfinite(arr)]
    dropped_inf = before_inf - len(arr)
    if dropped_inf > 0:
        print(f"  Dropped (Infinite values)   : {dropped_inf:,}")

    clean_n = len(arr)
    total_dropped = original_n - clean_n
    print(f"\n  Total dropped         : {total_dropped:,}")
    print(f"  Valid observations    : {clean_n:,}")

    # Hard stops
    if clean_n == 0:
        print("\n  [ERROR] No valid observations remain after cleaning.")
        return None

    if clean_n < 3:
        print(f"\n  [ERROR] Only {clean_n} valid observation(s) remain.")
        print("          Lilliefors Test requires n ≥ 3.")
        return None

    # Constant variable hard stop
    if is_effectively_constant(arr):
        print(f"\n  [ERROR] Column '{col}' has zero variance (all values are identical).")
        print("          Normality testing is meaningless for a constant variable.")
        print("          Please select a different column.")
        return None

    if clean_n < 10:
        print(f"\n  [WARNING] Small sample (n = {clean_n}).")
        print("            Lilliefors Test has low power for very small samples.")
        print("            Consider Shapiro-Wilk Test (normality_tests/shapiro_wilk_test.py)")
        print("            which is more powerful for n < 50.")

    print(f"\n  [OK] Data is clean and ready for Lilliefors Test.")
    return arr


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(arr: np.ndarray) -> dict:
    n = len(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
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


def display_descriptive_stats(desc: dict, col: str):
    print_section("DESCRIPTIVE STATISTICS")

    print(f"\n  Variable : {col}\n")

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

    val_w   = 20
    label_w = 30
    print(f"  {'Statistic':<{label_w}} {'Value':>{val_w}}")
    print(f"  {'-'*label_w} {'-'*val_w}")
    for name, key in stat_rows:
        print(f"  {name:<{label_w}} {fmt_val(desc[key]):>{val_w}}")

    # Skewness and kurtosis interpretation
    skew = desc["skewness"]
    kurt = desc["kurtosis"]
    print(f"\n  Skewness Interpretation:")
    if math.isnan(skew):
        print(f"    Not computable (insufficient data).")
    elif abs(skew) < 0.5:
        print(f"    Skewness = {skew:.4f} → Approximately symmetric.")
    elif abs(skew) < 1.0:
        direction = "right (positive)" if skew > 0 else "left (negative)"
        print(f"    Skewness = {skew:.4f} → Moderately skewed {direction}.")
    else:
        direction = "right (positive)" if skew > 0 else "left (negative)"
        print(f"    Skewness = {skew:.4f} → Highly skewed {direction}.")

    print(f"\n  Kurtosis Interpretation (excess):")
    if math.isnan(kurt):
        print(f"    Not computable (insufficient data).")
    elif abs(kurt) < 0.5:
        print(f"    Kurtosis = {kurt:.4f} → Mesokurtic (normal-like tails).")
    elif kurt > 0.5:
        print(f"    Kurtosis = {kurt:.4f} → Leptokurtic (heavy tails, more outliers).")
    else:
        print(f"    Kurtosis = {kurt:.4f} → Platykurtic (light tails, fewer outliers).")


# ======================================================================
# LILLIEFORS TEST
# ======================================================================

def run_lilliefors(arr: np.ndarray) -> dict | None:
    """
    Run Lilliefors Test using statsmodels.stats.diagnostic.lilliefors().

    The Lilliefors test is the KS test with Lilliefors-corrected critical
    values for the case where mean and SD are estimated from the sample.

    Parameters estimated: mean and standard deviation (normal distribution).

    Returns a result dictionary or None on failure.
    """
    n = len(arr)

    try:
        ks_stat, p_value = sm_lilliefors(arr, dist="norm", pvalmethod="table")
    except Exception as exc:
        print(f"  [ERROR] Lilliefors Test failed.\n          Details: {exc}")
        return None

    if math.isnan(ks_stat) or math.isnan(p_value):
        print("  [ERROR] Lilliefors Test returned NaN results.")
        print("          This may indicate insufficient variance in the data.")
        return None

    # p-value from statsmodels is bounded; flag when at boundary
    p_at_boundary = bool(p_value >= 0.20 or p_value <= 0.001)

    # Normal approximation decision using bool() — avoids numpy.bool_ identity issues
    is_normal = bool(p_value > 0.05)
    is_large_sample = bool(n >= LARGE_SAMPLE_THRESHOLD)

    # Estimated parameters used in the test
    est_mean = float(np.mean(arr))
    est_std  = float(np.std(arr, ddof=1))

    return {
        "ks_stat"        : float(ks_stat),
        "p_value"        : float(p_value),
        "n"              : n,
        "est_mean"       : est_mean,
        "est_std"        : est_std,
        "is_normal"      : is_normal,
        "is_large_sample": is_large_sample,
        "p_at_boundary"  : p_at_boundary,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_assumption_checks(arr: np.ndarray, col: str):
    print_section("ASSUMPTION CHECKS")

    n = len(arr)
    est_mean = float(np.mean(arr))
    est_std  = float(np.std(arr, ddof=1))

    print("\n  [1] Continuous Numeric Variable")
    print(f"      CONFIRMED: '{col}' is continuous and numeric.")
    print("      Lilliefors Test is appropriate for continuous variables.")
    print("      It is NOT designed for discrete or ordinal data.")

    print("\n  [2] Parameters Estimated from Sample")
    print(f"      CONFIRMED: Both mean and SD are unknown and estimated from data.")
    print(f"      Estimated Mean : {est_mean:,.4f}")
    print(f"      Estimated SD   : {est_std:,.4f}")
    print(f"      This is why Lilliefors (not standard KS) must be used.")
    print(f"      Standard KS test is anti-conservative when parameters are")
    print(f"      estimated from the same sample — it inflates p-values.")

    print("\n  [3] Why Lilliefors Test?")
    print("      The Lilliefors Test was selected because:")
    print("      - Population mean and SD are unknown (true for all real medical data).")
    print("      - Standard KS test requires pre-specified parameters.")
    print("      - Using estimated parameters in KS biases p-values upward,")
    print("        causing under-rejection of false normality.")
    print("      - Lilliefors (1967) derived corrected critical values for")
    print("        the case of estimated normal distribution parameters.")
    print("      - This produces a correctly-sized test at the stated α level.")

    print("\n  [4] Comparison with Other Normality Tests in VM-WIT-STATS")
    print(f"      Shapiro-Wilk     : Most powerful for n < 50. Recommended first.")
    print(f"      Anderson-Darling : Strong for n < 200. Sensitive to tails.")
    print(f"      Lilliefors       : Correct KS-based approach. Use for confirmation.")
    print(f"      Current n = {n:,}.")
    if n < 50:
        print(f"      [NOTE] For n = {n}, Shapiro-Wilk has higher power.")
        print(f"             Lilliefors result should be used as a cross-check.")
    elif n < 200:
        print(f"      [NOTE] For n = {n}, Anderson-Darling and Lilliefors are both appropriate.")
    else:
        print(f"      [NOTE] Large sample (n = {n:,}). See large-sample interpretation below.")

    print("\n  [5] Independence of Observations")
    print("      ASSUMED: Each observation is independent of all others.")
    print("      Lilliefors Test is not appropriate for time-series or clustered data.")


def display_test_results(results: dict, col: str):
    print_section("LILLIEFORS TEST RESULTS")

    sig    = not results["is_normal"]
    n      = results["n"]
    ks     = results["ks_stat"]
    p      = results["p_value"]

    print(f"\n  Test              : Lilliefors Test (Normality)")
    print(f"  Variable          : {col}")
    print(f"  Null Hypothesis   : Data follows a Normal distribution")
    print(f"                      (with mean and SD estimated from sample).")
    print(f"  Alt. Hypothesis   : Data does NOT follow a Normal distribution.")
    print(f"  Significance (α)  : 0.05")
    print(f"  Parameters        : Mean = {results['est_mean']:,.4f},  "
          f"SD = {results['est_std']:,.4f}  (estimated from data)\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Lilliefors D Statistic':<40} {ks:>20.4f}")
    print(f"  {'p-value':<40} {p:>20.4f}")
    print(f"  {'Sample Size (n)':<40} {n:>20,}")
    print(f"  {'Estimated Mean':<40} {results['est_mean']:>20,.4f}")
    print(f"  {'Estimated SD':<40} {results['est_std']:>20,.4f}")
    print(f"  {SEPARATOR2[:68]}")

    if results["p_at_boundary"]:
        print(f"\n  [NOTE] p-value is at the boundary of the Lilliefors table")
        print(f"         (p ≤ 0.001 or p ≥ 0.20). The reported value is the")
        print(f"         closest tabulated bound. Interpret accordingly.")

    verdict = "FAIL TO REJECT NORMALITY (p > 0.05)" if results["is_normal"] \
              else "REJECT NORMALITY (p ≤ 0.05)"
    print(f"\n  Result  : {verdict}")
    print(f"  p-value : {p:.4f}  (α = 0.05)")


def display_large_sample_note(results: dict):
    """Display large-sample interpretation when n >= threshold."""
    if not results["is_large_sample"]:
        return

    print_section("LARGE SAMPLE INTERPRETATION")

    n = results["n"]
    print(f"\n  [IMPORTANT] Large Sample Warning (n = {n:,})")
    print(f"\n  With large samples, the Lilliefors Test — like all formal")
    print(f"  normality tests — gains power to detect trivially small deviations")
    print(f"  from normality that are statistically detectable but practically")
    print(f"  irrelevant for the purpose of selecting a statistical test.")
    print(f"\n  This means:")
    print(f"    • A significant result (p ≤ 0.05) at n = {n:,} does NOT")
    print(f"      automatically mean parametric tests are inappropriate.")
    print(f"    • Parametric tests (T-Test, ANOVA) are robust to minor normality")
    print(f"      violations, especially with large samples (Central Limit Theorem).")
    print(f"    • The key question is: how large is the deviation from normality,")
    print(f"      not merely whether it is statistically detectable.")
    print(f"\n  Guidance for large samples:")
    print(f"    1. Examine the D statistic magnitude, not just p-value.")
    print(f"    2. Inspect skewness and kurtosis (see Descriptive Statistics).")
    print(f"    3. Consider visual inspection (histogram, Q-Q plot).")
    print(f"    4. For n > 100, parametric tests are generally robust to")
    print(f"       moderate non-normality due to the Central Limit Theorem.")
    print(f"    5. Use non-parametric tests when clinical data is ordinal")
    print(f"       or when extreme skewness / outliers are present.")


def display_effect_magnitude(results: dict):
    """
    Interpret the practical magnitude of the D statistic.
    Unlike p-value, D statistic is not affected by sample size in the
    same way — larger D indicates a greater departure from normality.
    """
    print_section("D STATISTIC INTERPRETATION")

    d = results["ks_stat"]
    n = results["n"]

    print(f"\n  Lilliefors D Statistic : {d:.4f}")
    print(f"  The D statistic is the maximum absolute difference between")
    print(f"  the empirical CDF and the fitted Normal CDF.")
    print(f"  It measures the magnitude of departure from normality,")
    print(f"  independent of sample size.\n")

    print(f"  D Magnitude Interpretation:")
    if d < 0.05:
        magnitude = "Very small — data closely follows a Normal distribution."
    elif d < 0.10:
        magnitude = "Small — minor deviation from Normality."
    elif d < 0.15:
        magnitude = "Moderate — noticeable departure from Normality."
    elif d < 0.20:
        magnitude = "Large — substantial departure from Normality."
    else:
        magnitude = "Very large — strong evidence of non-Normal distribution."

    print(f"  D = {d:.4f} → {magnitude}")

    if results["is_large_sample"] and not results["is_normal"]:
        print(f"\n  [NOTE] At n = {n:,}, even small D values can produce p ≤ 0.05.")
        print(f"         The D statistic magnitude is more informative than the p-value")
        print(f"         for practical decision-making at large sample sizes.")


def display_test_selection(results: dict, desc: dict, col: str):
    print_section("RECOMMENDED STATISTICAL TESTS")

    is_normal = results["is_normal"]
    is_large  = results["is_large_sample"]
    n         = results["n"]
    skew      = desc["skewness"]
    d         = results["ks_stat"]

    # For large samples: refine recommendation based on D statistic magnitude
    # and skewness, not just p-value
    if is_large and not is_normal:
        practical_normal = bool(d < 0.10 and abs(skew) < 1.0)
    else:
        practical_normal = is_normal

    print(f"\n  Variable   : {col}")
    print(f"  n          : {n:,}")
    print(f"  Normality  : {'Supported' if is_normal else 'Not supported'} (p {'>' if is_normal else '≤'} 0.05)")
    if is_large and not is_normal:
        print(f"  D Stat     : {d:.4f}  ({'Minor' if d < 0.10 else 'Substantial'} deviation)")
        print(f"  Skewness   : {fmt_val(skew)}")
        print(f"  Practical  : {'Minor deviation — parametric tests may still be appropriate' if practical_normal else 'Substantial deviation — consider non-parametric tests'}")

    print(f"\n  {'─'*60}")

    if practical_normal:
        print(f"\n  PARAMETRIC TESTS ARE APPROPRIATE:\n")
        print(f"    Two-Group Comparison   : Independent T-Test")
        print(f"    Paired Comparison      : Paired T-Test")
        print(f"    Multiple Groups        : One-Way ANOVA")
        print(f"    Correlation            : Pearson Correlation")
        print(f"    Regression             : Linear Regression")
        print(f"\n  NON-PARAMETRIC ALTERNATIVES (always valid):\n")
        print(f"    Two-Group Comparison   : Mann-Whitney U Test")
        print(f"    Paired Comparison      : Wilcoxon Signed-Rank Test")
        print(f"    Multiple Groups        : Kruskal-Wallis Test")
        print(f"    Repeated Measures      : Friedman Test")
        print(f"    Correlation            : Spearman Correlation")
    else:
        print(f"\n  NON-PARAMETRIC TESTS ARE RECOMMENDED:\n")
        print(f"    Two-Group Comparison   : Mann-Whitney U Test")
        print(f"    Paired Comparison      : Wilcoxon Signed-Rank Test")
        print(f"    Multiple Groups        : Kruskal-Wallis Test")
        print(f"    Repeated Measures      : Friedman Test")
        print(f"    Correlation            : Spearman Correlation")
        print(f"\n  PARAMETRIC TESTS (use cautiously):\n")
        print(f"    T-Tests and ANOVA are robust to moderate non-normality,")
        print(f"    especially with n > 30 (Central Limit Theorem applies).")
        if is_large:
            print(f"    With n = {n:,}, parametric tests may still be appropriate")
            print(f"    if the violation is minor (small D statistic, low skewness).")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            col: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    p       = results["p_value"]
    d       = results["ks_stat"]
    n       = results["n"]
    is_norm = results["is_normal"]
    is_lg   = results["is_large_sample"]
    skew    = desc["skewness"]
    kurt    = desc["kurtosis"]

    print(f"\n  VARIABLE          : {col}")
    print(f"  SAMPLE SIZE       : {n:,}")
    print(f"  TEST              : Lilliefors Test")
    print(f"  ESTIMATED MEAN    : {results['est_mean']:,.4f}")
    print(f"  ESTIMATED SD      : {results['est_std']:,.4f}")

    print(f"\n  WHY LILLIEFORS TEST?")
    print(f"  In all real medical datasets, the population mean and standard")
    print(f"  deviation are unknown and must be estimated from the sample.")
    print(f"  The standard Kolmogorov-Smirnov test requires pre-specified")
    print(f"  parameters — when estimated parameters are used instead, the")
    print(f"  test becomes anti-conservative, inflating p-values and under-")
    print(f"  detecting non-normality. The Lilliefors correction (1967) adjusts")
    print(f"  the critical values for parameter estimation, producing a correctly-")
    print(f"  sized test that is statistically appropriate for medical datasets.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if is_norm:
        print(f"  The Lilliefors Test does NOT provide sufficient evidence against")
        print(f"  normality for '{col}'.")
        print(f"  (D = {d:.4f}, p = {p:.4f})")
        print(f"  We fail to reject the null hypothesis that the data follows")
        print(f"  a Normal distribution.")
        print(f"\n  This does NOT prove the data is normally distributed.")
        print(f"  It indicates that the available evidence is insufficient")
        print(f"  to reject normality at the 0.05 significance level.")
    else:
        print(f"  The Lilliefors Test provides evidence against normality")
        print(f"  for '{col}'.")
        print(f"  (D = {d:.4f}, p = {p:.4f})")
        print(f"  We reject the null hypothesis of normality at α = 0.05.")
        if is_lg:
            print(f"\n  Large-sample context: With n = {n:,}, the test has high")
            print(f"  power to detect small deviations. The D statistic magnitude")
            print(f"  (D = {d:.4f}) and skewness ({fmt_val(skew)}) should be")
            print(f"  considered alongside the p-value to assess practical impact.")

    print(f"\n  D STATISTIC MAGNITUDE:")
    if d < 0.05:
        print(f"  D = {d:.4f} → Very small departure. Data closely approximates Normal.")
    elif d < 0.10:
        print(f"  D = {d:.4f} → Small departure from Normality.")
    elif d < 0.15:
        print(f"  D = {d:.4f} → Moderate departure. Non-normality is noticeable.")
    elif d < 0.20:
        print(f"  D = {d:.4f} → Large departure. Non-normality is substantial.")
    else:
        print(f"  D = {d:.4f} → Very large departure. Strong non-normality present.")

    print(f"\n  DISTRIBUTIONAL CHARACTERISTICS:")
    if not math.isnan(skew):
        if abs(skew) < 0.5:
            print(f"  Skewness = {skew:.4f}: Distribution is approximately symmetric.")
        elif abs(skew) < 1.0:
            direction = "right" if skew > 0 else "left"
            print(f"  Skewness = {skew:.4f}: Moderate {direction}-skew detected.")
        else:
            direction = "right" if skew > 0 else "left"
            print(f"  Skewness = {skew:.4f}: Strong {direction}-skew. Non-normality likely.")
    if not math.isnan(kurt):
        if abs(kurt) < 0.5:
            print(f"  Kurtosis = {kurt:.4f}: Normal-like tail behaviour.")
        elif kurt > 0.5:
            print(f"  Kurtosis = {kurt:.4f}: Heavy tails. Outliers likely present.")
        else:
            print(f"  Kurtosis = {kurt:.4f}: Light tails relative to Normal.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Normality is one of several assumptions underlying parametric tests.")
    print(f"  In clinical research, the choice between parametric and non-parametric")
    print(f"  methods should consider: sample size, degree of non-normality,")
    print(f"  measurement scale, presence of outliers, and research objectives.")
    print(f"  A non-significant Lilliefors result supports — but does not prove —")
    print(f"  that parametric tests are appropriate. A significant result warrants")
    print(f"  consideration of non-parametric alternatives, especially in small samples.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Lilliefors Test is less powerful than Shapiro-Wilk for n < 50.")
    print(f"  • All formal normality tests lose interpretive value at large n")
    print(f"    because trivial deviations become statistically detectable.")
    print(f"  • The test assesses global normality; local deviations in tails")
    print(f"    may be missed or overemphasised depending on sample composition.")
    print(f"  • p-values from statsmodels are bounded by the tabulated range")
    print(f"    (approximately 0.001 to 0.20). Values outside this range are")
    print(f"    reported as boundary values.")
    print(f"  • This module tests for normality only. It does not test equal")
    print(f"    variance, independence, or other statistical assumptions.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 col: str,
                 dataset_path: str) -> str:
    p       = results["p_value"]
    d       = results["ks_stat"]
    n       = results["n"]
    is_norm = results["is_normal"]
    is_lg   = results["is_large_sample"]
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Lilliefors Test (Normality)")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    lines.append(f"  Variable : {col}")
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
    label_w = 30
    val_w   = 20
    lines.append(f"  {'Statistic':<{label_w}} {'Value':>{val_w}}")
    lines.append(f"  {'-'*label_w} {'-'*val_w}")
    for name, key in stat_rows:
        lines.append(f"  {name:<{label_w}} {fmt_val(desc[key]):>{val_w}}")

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append(f"  [1] Continuous numeric variable   : CONFIRMED")
    lines.append(f"  [2] Parameters estimated from sample : CONFIRMED")
    lines.append(f"      Estimated Mean : {results['est_mean']:,.4f}")
    lines.append(f"      Estimated SD   : {results['est_std']:,.4f}")
    lines.append(f"  [3] Standard KS not used          : CORRECT (parameters estimated)")
    lines.append(f"      Standard KS inflates p-values when parameters are estimated.")
    lines.append(f"      Lilliefors corrected critical values used instead.")
    lines.append(f"  [4] Independence of observations   : ASSUMED")
    lines.append(f"  [5] Sample size                   : n = {n:,}")
    if n < 50:
        lines.append(f"       NOTE: Shapiro-Wilk has higher power for n < 50.")
    elif n < 200:
        lines.append(f"       NOTE: Anderson-Darling and Lilliefors are both appropriate.")
    else:
        lines.append(f"       NOTE: Large sample — see large-sample interpretation.")

    # Test Results
    lines.append("")
    lines.append("  LILLIEFORS TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test                    : Lilliefors Test (Normality)")
    lines.append(f"  Variable                : {col}")
    lines.append(f"  Null Hypothesis         : Data follows Normal distribution")
    lines.append(f"                            (mean and SD estimated from sample)")
    lines.append(f"  Alt. Hypothesis         : Data does NOT follow Normal distribution")
    lines.append(f"  Lilliefors D Statistic  : {d:.4f}")
    lines.append(f"  p-value                 : {p:.4f}")
    lines.append(f"  Sample Size (n)         : {n:,}")
    lines.append(f"  Estimated Mean          : {results['est_mean']:,.4f}")
    lines.append(f"  Estimated SD            : {results['est_std']:,.4f}")
    if results["p_at_boundary"]:
        lines.append(f"  p-value Boundary Note   : p is at Lilliefors table boundary.")
        lines.append(f"                            Reported value is closest tabulated bound.")
    verdict = "FAIL TO REJECT NORMALITY (p > 0.05)" if is_norm \
              else "REJECT NORMALITY (p ≤ 0.05)"
    lines.append(f"  Result                  : {verdict}")

    # D Statistic Magnitude
    lines.append("")
    lines.append("  D STATISTIC MAGNITUDE")
    lines.append(SEPARATOR2)
    if d < 0.05:
        d_interp = "Very small — data closely follows a Normal distribution."
    elif d < 0.10:
        d_interp = "Small — minor deviation from Normality."
    elif d < 0.15:
        d_interp = "Moderate — noticeable departure from Normality."
    elif d < 0.20:
        d_interp = "Large — substantial departure from Normality."
    else:
        d_interp = "Very large — strong evidence of non-Normal distribution."
    lines.append(f"  D = {d:.4f} → {d_interp}")

    # Large sample note
    if is_lg:
        lines.append("")
        lines.append("  LARGE SAMPLE INTERPRETATION")
        lines.append(SEPARATOR2)
        lines.append(f"  n = {n:,}. Large samples give the Lilliefors Test high power to")
        lines.append("  detect trivially small deviations that are statistically significant")
        lines.append("  but practically irrelevant for test selection.")
        lines.append("  A rejected normality test at large n does NOT automatically mean")
        lines.append("  parametric tests are inappropriate.")
        lines.append("  Consider: D statistic magnitude, skewness, kurtosis, and the")
        lines.append("  Central Limit Theorem (robust for parametric tests at large n).")

    # Recommended tests
    lines.append("")
    lines.append("  RECOMMENDED STATISTICAL TESTS")
    lines.append(SEPARATOR2)
    skew = desc["skewness"]
    if is_lg and not is_norm:
        practical_normal = bool(d < 0.10 and abs(skew) < 1.0)
    else:
        practical_normal = is_norm

    if practical_normal:
        lines.append("  Normality supported. Parametric tests are appropriate.")
        lines.append("    Two-Group Comparison   : Independent T-Test")
        lines.append("    Paired Comparison      : Paired T-Test")
        lines.append("    Multiple Groups        : One-Way ANOVA")
        lines.append("    Correlation            : Pearson Correlation")
        lines.append("    Regression             : Linear Regression")
        lines.append("  Non-parametric alternatives are always valid alternatives.")
    else:
        lines.append("  Non-normality detected. Non-parametric tests recommended.")
        lines.append("    Two-Group Comparison   : Mann-Whitney U Test")
        lines.append("    Paired Comparison      : Wilcoxon Signed-Rank Test")
        lines.append("    Multiple Groups        : Kruskal-Wallis Test")
        lines.append("    Repeated Measures      : Friedman Test")
        lines.append("    Correlation            : Spearman Correlation")
        if is_lg:
            lines.append(f"  With n = {n:,}, parametric tests may still be appropriate")
            lines.append("  if deviation is minor (small D, low skewness).")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if is_norm:
        lines.append("  The Lilliefors Test does not provide sufficient evidence against normality.")
        lines.append(f"  D = {d:.4f}, p = {p:.4f}. Fail to reject H0.")
    else:
        lines.append("  The Lilliefors Test provides evidence against normality.")
        lines.append(f"  D = {d:.4f}, p = {p:.4f}. Reject H0 at α = 0.05.")
        if is_lg:
            lines.append(f"  Large sample (n = {n:,}): interpret D magnitude alongside p-value.")
    lines.append("")
    lines.append("  The Lilliefors Test uses corrected critical values for the case")
    lines.append("  where mean and SD are estimated from the sample. Standard KS")
    lines.append("  must not be used for normality testing in this scenario.")
    lines.append("  Statistical significance does not automatically indicate that")
    lines.append("  parametric tests are inappropriate — sample size, D magnitude,")
    lines.append("  and distributional characteristics should all be considered.")

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

def run_lilliefors_test():
    print_header()

    if not check_statsmodels():
        sys.exit(1)

    print("  This module performs the Lilliefors Test for Normality.")
    print("  It is the statistically correct KS-based normality test when")
    print("  population mean and SD are unknown (estimated from the sample).")
    print("  Standard KS test must NOT be used in this scenario.")

    # Step 1: Load CSV
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select variable
    col = select_numeric_column(df)
    if col is None:
        sys.exit(1)

    # Step 4: Clean data
    arr = clean_data(df, col)
    if arr is None:
        sys.exit(1)

    # Step 5: Descriptive statistics
    desc = compute_descriptive_stats(arr)
    display_descriptive_stats(desc, col)

    # Step 6: Assumption checks
    display_assumption_checks(arr, col)

    # Step 7: Run Lilliefors Test
    print_section("RUNNING LILLIEFORS TEST")
    results = run_lilliefors(arr)
    if results is None:
        print("\n  [FATAL] Lilliefors Test could not be completed.")
        sys.exit(1)

    # Step 8: Display results
    display_test_results(results, col)
    display_large_sample_note(results)
    display_effect_magnitude(results)
    display_test_selection(results, desc, col)
    display_interpretation(results, desc, col)

    # Step 9: Build and save report
    report_text = build_report(results, desc, col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Lilliefors Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_lilliefors_test()