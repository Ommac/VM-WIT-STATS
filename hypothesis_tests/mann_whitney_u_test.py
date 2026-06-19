
import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import mannwhitneyu, norm

warnings.filterwarnings("ignore")


SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "mann_whitney_u_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Mann-Whitney U Test
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


def safe_float(value) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_effectively_constant(series: pd.Series, tol: float = 1e-10) -> bool:
    """Return True if all values in the series are effectively the same."""
    return bool((series.max() - series.min()) < tol)


def rank_biserial_correlation(u_stat: float, n1: int, n2: int) -> float:
    """
    Compute the rank-biserial correlation (r) as an effect size for
    the Mann-Whitney U Test.

        r = 1 - (2 * U) / (n1 * n2)

    Returns a value in [-1, 1].
    """
    denom = n1 * n2
    if denom == 0:
        return float("nan")
    return 1.0 - (2.0 * u_stat) / denom


def interpret_effect_size(r: float) -> str:
    """
    Interpret rank-biserial correlation using Cohen (1992) benchmarks:
      |r| < 0.10  -> Negligible
      |r| < 0.30  -> Small
      |r| < 0.50  -> Medium
      |r| >= 0.50 -> Large
    """
    abs_r = abs(r)
    if abs_r < 0.10:
        return "Negligible"
    elif abs_r < 0.30:
        return "Small"
    elif abs_r < 0.50:
        return "Medium"
    else:
        return "Large"


def ci_median_difference(group1: np.ndarray,
                         group2: np.ndarray,
                         confidence: float = 0.95) -> tuple:
    """
    Estimate 95% CI for the Hodges-Lehmann estimator (pseudo-median of
    pairwise differences between the two groups).

    Returns (lower_bound, upper_bound, hl_estimate).
    """
    try:
        diffs = np.subtract.outer(group1, group2).flatten()
        hl_estimate = float(np.median(diffs))
        alpha = 1.0 - confidence
        z = norm.ppf(1.0 - alpha / 2.0)
        n1, n2 = len(group1), len(group2)
        total_pairs = n1 * n2
        # Normal approximation for CI on Wilcoxon rank-sum
        sigma = math.sqrt(total_pairs * (n1 + n2 + 1) / 12.0)
        margin = z * sigma / total_pairs
        lower = hl_estimate - margin
        upper = hl_estimate + margin
        return lower, upper, hl_estimate
    except Exception:
        return float("nan"), float("nan"), float("nan")


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    """Prompt user for a CSV file path and validate its existence."""
    print_section("STEP 1 | LOAD DATASET")
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
    """Safely load a CSV file with comprehensive error handling."""
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
    """Display all column names with their dtypes."""
    print_section("STEP 2 | AVAILABLE COLUMNS")
    print(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15}")
    print(f"  {'-'*5} {'-'*40} {'-'*15}")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        print(f"  {idx:<5} {col:<40} {str(dtype):<15}")


# ======================================================================
# COLUMN SELECTION & VALIDATION
# ======================================================================

def select_outcome_column(df: pd.DataFrame) -> str | None:
    """
    Prompt user to select the outcome (numeric continuous) column.
    Validates that the column exists and is or can be treated as numeric.
    """
    print_section("STEP 3 | SELECT OUTCOME VARIABLE")
    print("\n  The outcome variable must be a continuous numeric column.")
    print("  Example: charges, meanbp, age, temperature, etc.")

    while True:
        col = input("\n  Enter outcome column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

        # Attempt coercion to numeric
        coerced = pd.to_numeric(df[col], errors="coerce")
        valid_count = coerced.notna().sum()
        total_count = len(coerced)

        if valid_count == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values after coercion.")
            print("          Please choose a different column.")
            continue

        non_numeric = total_count - valid_count
        if non_numeric > 0:
            print(f"  [WARNING] {non_numeric:,} non-numeric value(s) in '{col}' will be treated as missing.")

        print(f"  [OK] Outcome column selected: '{col}'")
        print(f"       Valid numeric values: {valid_count:,} / {total_count:,}")
        return col


def select_grouping_column(df: pd.DataFrame, outcome_col: str) -> str | None:
    """
    Prompt user to select the grouping column.
    Validates exactly 2 unique groups exist after cleaning.
    """
    print_section("STEP 4 | SELECT GROUPING VARIABLE")
    print("\n  The grouping variable must have EXACTLY 2 unique groups.")
    print("  Example: sex (Male/Female), death (0/1), treatment (A/B), etc.")

    while True:
        col = input("\n  Enter grouping column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue
        if col == outcome_col:
            print("  [ERROR] Grouping column cannot be the same as the outcome column.")
            continue

        # Drop rows where grouping column is NaN
        valid_groups = df[col].dropna().unique()
        n_groups = len(valid_groups)

        if n_groups < 2:
            print(f"  [ERROR] Column '{col}' has fewer than 2 unique groups ({n_groups} found).")
            print("          Mann-Whitney U Test requires exactly 2 groups.")
            continue
        if n_groups > 2:
            print(f"  [ERROR] Column '{col}' has {n_groups} unique groups: {list(valid_groups)}")
            print("          Mann-Whitney U Test requires exactly 2 groups.")
            print("          Consider using Kruskal-Wallis Test for 3+ groups.")
            continue

        print(f"  [OK] Grouping column selected: '{col}'")
        print(f"       Groups found: {list(valid_groups)}")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               outcome_col: str,
               group_col: str) -> pd.DataFrame | None:
    """
    Perform comprehensive data cleaning:
      1. Select only relevant columns.
      2. Coerce outcome to numeric.
      3. Drop rows with NaN in either column.
      4. Drop rows with infinite values in outcome.
      5. Re-check group validity after cleaning.
    """
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    working = df[[outcome_col, group_col]].copy()
    original_rows = len(working)
    print(f"\n  Original rows : {original_rows:,}")

    # Coerce outcome to numeric
    working[outcome_col] = pd.to_numeric(working[outcome_col], errors="coerce")

    # Drop rows where grouping variable is NaN
    before = len(working)
    working = working.dropna(subset=[group_col])
    dropped_group_nan = before - len(working)
    if dropped_group_nan > 0:
        print(f"  Dropped (NaN in group column)   : {dropped_group_nan:,} row(s)")

    # Drop rows where outcome is NaN
    before = len(working)
    working = working.dropna(subset=[outcome_col])
    dropped_outcome_nan = before - len(working)
    if dropped_outcome_nan > 0:
        print(f"  Dropped (NaN in outcome column) : {dropped_outcome_nan:,} row(s)")

    # Drop rows where outcome is infinite
    before = len(working)
    inf_mask = np.isinf(working[outcome_col])
    working = working[~inf_mask]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        print(f"  Dropped (Infinite outcome values): {dropped_inf:,} row(s)")

    clean_rows = len(working)
    total_dropped = original_rows - clean_rows
    print(f"\n  Total rows dropped : {total_dropped:,}")
    print(f"  Rows for analysis  : {clean_rows:,}")

    if clean_rows == 0:
        print("\n  [ERROR] No valid rows remain after cleaning.")
        print("          Cannot proceed with analysis.")
        return None

    # Re-check groups after cleaning
    unique_groups = working[group_col].unique()
    n_groups = len(unique_groups)

    if n_groups < 2:
        print(f"\n  [ERROR] After cleaning, fewer than 2 groups remain ({n_groups} found).")
        print("          Cannot proceed with Mann-Whitney U Test.")
        return None
    if n_groups > 2:
        print(f"\n  [ERROR] After cleaning, more than 2 groups remain ({n_groups} found).")
        print(f"          Groups: {list(unique_groups)}")
        print("          Please check your grouping variable.")
        return None

    print(f"\n  [OK] Data is clean and ready for analysis.")
    return working


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(group_data: dict) -> dict:
    """
    Compute descriptive statistics for each group.
    Returns a dictionary keyed by group label.
    """
    stats_dict = {}
    for label, values in group_data.items():
        arr = np.array(values, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        stats_dict[label] = {
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
    return stats_dict


def display_descriptive_stats(desc: dict, outcome_col: str, group_col: str):
    """Pretty-print descriptive statistics for both groups."""
    print_section("DESCRIPTIVE STATISTICS")
    labels = list(desc.keys())

    print(f"\n  Outcome Variable : {outcome_col}")
    print(f"  Grouping Variable: {group_col}\n")

    header_fmt = "  {:<28} {:>18} {:>18}"
    row_fmt    = "  {:<28} {:>18} {:>18}"

    print(header_fmt.format("Statistic", str(labels[0]), str(labels[1])))
    print(f"  {'-'*28} {'-'*18} {'-'*18}")

    def fmt(val):
        if isinstance(val, float) and math.isnan(val):
            return "N/A"
        return f"{val:,.4f}" if isinstance(val, float) else str(val)

    rows = [
        ("Sample Size (n)",    "n"),
        ("Mean",               "mean"),
        ("Median",             "median"),
        ("Std Deviation",      "std"),
        ("Std Error of Mean",  "sem"),
        ("Minimum",            "min"),
        ("Maximum",            "max"),
        ("25th Percentile",    "q1"),
        ("75th Percentile",    "q3"),
        ("IQR (Q3 - Q1)",      "iqr"),
        ("Skewness",           "skewness"),
        ("Kurtosis",           "kurtosis"),
    ]
    for display_name, key in rows:
        v0 = fmt(desc[labels[0]][key])
        v1 = fmt(desc[labels[1]][key])
        print(row_fmt.format(display_name, v0, v1))


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def check_normality(group_data: dict) -> dict:
    """
    Run Shapiro-Wilk normality test on each group.
    Returns a dict with statistic, p-value, and normality verdict.
    """
    results = {}
    for label, values in group_data.items():
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n < 3:
            results[label] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": False, "note": "Too few samples (n < 3)"
            }
        elif n > 5000:
            # Shapiro-Wilk is unreliable for very large samples; skip
            results[label] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": False,
                "note": "n > 5000: Shapiro-Wilk not computed (large sample)"
            }
        else:
            stat, p = stats.shapiro(arr)
            results[label] = {
                "stat": float(stat), "p": float(p),
                "normal": p > 0.05,
                "note": ""
            }
    return results


def display_assumption_checks(normality: dict, group_data: dict):
    """Display assumption checks relevant to Mann-Whitney U Test."""
    print_section("ASSUMPTION CHECKS")

    print("\n  [1] Independence of Observations")
    print("      ASSUMED: Each observation belongs to one group only.")
    print("      Ensure data collection followed independent sampling.")

    print("\n  [2] Normality of Outcome Variable (Shapiro-Wilk Test)")
    print(f"      H0: Data is normally distributed  (p > 0.05 = Normal)")
    print(f"      H1: Data is NOT normally distributed (p ≤ 0.05 = Non-Normal)\n")

    for label, res in normality.items():
        n = len(group_data[label])
        print(f"      Group: {label}  (n = {n:,})")
        if res["note"]:
            print(f"        Note   : {res['note']}")
        else:
            verdict = "Normal" if res["normal"] else "Non-Normal"
            print(f"        W Stat : {res['stat']:.4f}")
            print(f"        p-value: {res['p']:.4f}")
            print(f"        Result : {verdict}")
        print()

    print("  [3] Scale of Measurement")
    print("      ASSUMED: Outcome variable is at least ordinal scale.")
    print("      Continuous and ordinal data are both appropriate.")

    print("\n  [4] Shape of Distribution (Stochastic Dominance)")
    print("      Mann-Whitney U Test does NOT require equal variances.")
    print("      It tests whether one group tends to have larger values.")

    print("\n  [CONCLUSION] Mann-Whitney U Test Selection Rationale:")
    print("      The Mann-Whitney U Test was selected because:")
    print("      - It is the non-parametric alternative to the Independent T-Test.")
    print("      - It does not assume normally distributed data.")
    print("      - It is robust to outliers and skewed distributions.")
    print("      - It compares rank-based distributions between two groups.")
    print("      - It is appropriate for continuous or ordinal outcome variables.")


# ======================================================================
# MANN-WHITNEY U TEST
# ======================================================================

def run_mann_whitney(group1: np.ndarray,
                     group2: np.ndarray,
                     label1,
                     label2) -> dict | None:
    """
    Execute Mann-Whitney U Test using scipy.stats.mannwhitneyu.
    Uses two-sided alternative hypothesis by default.

    Returns a dictionary of all computed statistics, or None on failure.
    """
    n1, n2 = len(group1), len(group2)

    # Guard: minimum sample size
    if n1 < 1 or n2 < 1:
        print("  [ERROR] One or both groups are empty after cleaning.")
        return None
    if n1 < 2 or n2 < 2:
        print(f"  [WARNING] Very small sample size detected: "
              f"Group '{label1}' n={n1}, Group '{label2}' n={n2}.")
        print("            Results may be unreliable. Proceeding with caution.")

    # Guard: constant groups
    if is_effectively_constant(pd.Series(group1)):
        print(f"  [WARNING] Group '{label1}' has zero variance (all values identical).")
    if is_effectively_constant(pd.Series(group2)):
        print(f"  [WARNING] Group '{label2}' has zero variance (all values identical).")

    try:
        u_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    except ValueError as exc:
        print(f"  [ERROR] Mann-Whitney U Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Mann-Whitney U Test.\n          Details: {exc}")
        return None

    # Check for NaN results
    if math.isnan(u_stat) or math.isnan(p_value):
        print("  [ERROR] Mann-Whitney U Test returned NaN results.")
        print("          This may indicate insufficient data or constant values.")
        return None

    # U statistic for the second group
    u_stat_g2 = (n1 * n2) - u_stat

    # Effect size: rank-biserial correlation
    r = rank_biserial_correlation(u_stat, n1, n2)
    effect_label = interpret_effect_size(r) if not math.isnan(r) else "N/A"

    # Hodges-Lehmann estimator and 95% CI for median difference
    ci_lower, ci_upper, hl_estimate = ci_median_difference(group1, group2)

    # Z-score approximation (for large samples)
    total_pairs = n1 * n2
    mean_u = total_pairs / 2.0
    std_u = math.sqrt(total_pairs * (n1 + n2 + 1) / 12.0) if (n1 + n2 + 1) > 0 else float("nan")
    z_score = (u_stat - mean_u) / std_u if (not math.isnan(std_u) and std_u != 0) else float("nan")

    return {
        "u_stat"       : float(u_stat),
        "u_stat_g2"    : float(u_stat_g2),
        "p_value"      : float(p_value),
        "n1"           : n1,
        "n2"           : n2,
        "total_n"      : n1 + n2,
        "r"            : float(r),
        "effect_label" : effect_label,
        "ci_lower"     : ci_lower,
        "ci_upper"     : ci_upper,
        "hl_estimate"  : hl_estimate,
        "z_score"      : z_score,
        "mean_u"       : mean_u,
        "std_u"        : std_u,
        "label1"       : label1,
        "label2"       : label2,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, desc: dict, outcome_col: str):
    """Display the core Mann-Whitney U Test results."""
    print_section("MANN-WHITNEY U TEST RESULTS")

    l1, l2 = results["label1"], results["label2"]
    sig = results["p_value"] < 0.05

    print(f"\n  Test             : Mann-Whitney U Test (Two-Tailed)")
    print(f"  Null Hypothesis  : The distributions of '{outcome_col}' are")
    print(f"                     equal across both groups.")
    print(f"  Alt. Hypothesis  : The distributions differ between groups.")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<35} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'U Statistic (Group: ' + str(l1) + ')':<35} {results['u_stat']:>20,.4f}")
    print(f"  {'U Statistic (Group: ' + str(l2) + ')':<35} {results['u_stat_g2']:>20,.4f}")
    print(f"  {'Z Score (approx.)':<35} {results['z_score']:>20.4f}")
    print(f"  {'p-value (two-sided)':<35} {results['p_value']:>20.4f}")
    print(f"  {'Sample Size (Group: ' + str(l1) + ')':<35} {results['n1']:>20,}")
    print(f"  {'Sample Size (Group: ' + str(l2) + ')':<35} {results['n2']:>20,}")
    print(f"  {'Total Sample Size':<35} {results['total_n']:>20,}")
    print(f"  {'Median (Group: ' + str(l1) + ')':<35} {desc[l1]['median']:>20,.4f}")
    print(f"  {'Median (Group: ' + str(l2) + ')':<35} {desc[l2]['median']:>20,.4f}")
    print(f"  {'Mean (Group: ' + str(l1) + ')':<35} {desc[l1]['mean']:>20,.4f}")
    print(f"  {'Mean (Group: ' + str(l2) + ')':<35} {desc[l2]['mean']:>20,.4f}")
    print(f"  {SEPARATOR2[:68]}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result           : {sig_str}")
    print(f"  p-value          : {results['p_value']:.4f}  (α = 0.05)")


def display_effect_size(results: dict):
    """Display effect size and Hodges-Lehmann CI."""
    print_section("EFFECT SIZE & CONFIDENCE INTERVAL")

    r = results["r"]
    hl = results["hl_estimate"]
    lo = results["ci_lower"]
    hi = results["ci_upper"]

    print(f"\n  Effect Size Measure : Rank-Biserial Correlation (r)")
    print(f"  Formula             : r = 1 - (2U) / (n1 × n2)\n")

    print(f"  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'Rank-Biserial Correlation (r)':<40} {r:>20.4f}")
    print(f"  {'Effect Size Interpretation':<40} {results['effect_label']:>20}")
    print(f"\n  {'Hodges-Lehmann Estimator':<40} {hl:>20.4f}")
    print(f"  {'95% CI Lower Bound':<40} {lo:>20.4f}")
    print(f"  {'95% CI Upper Bound':<40} {hi:>20.4f}")

    print(f"\n  Effect Size Benchmarks (Cohen 1992):")
    print(f"    |r| < 0.10 → Negligible")
    print(f"    |r| < 0.30 → Small")
    print(f"    |r| < 0.50 → Medium")
    print(f"    |r| ≥ 0.50 → Large")

    print(f"\n  Hodges-Lehmann Estimator:")
    print(f"    This is the pseudo-median of all pairwise differences")
    print(f"    (Group {results['label1']} − Group {results['label2']}).")
    print(f"    Value = {hl:.4f} indicates the typical shift between groups.")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            normality: dict,
                            outcome_col: str,
                            group_col: str):
    """Generate and display a full medical-grade statistical interpretation."""
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    l1, l2 = results["label1"], results["label2"]
    p = results["p_value"]
    r = results["r"]
    sig = p < 0.05

    med1 = desc[l1]["median"]
    med2 = desc[l2]["median"]
    higher_group = l1 if med1 >= med2 else l2
    lower_group  = l2 if med1 >= med2 else l1
    higher_med   = max(med1, med2)
    lower_med    = min(med1, med2)

    print(f"\n  OUTCOME VARIABLE : {outcome_col}")
    print(f"  GROUPING VARIABLE: {group_col}")
    print(f"  GROUP 1          : {l1}  (n = {results['n1']:,})")
    print(f"  GROUP 2          : {l2}  (n = {results['n2']:,})")

    print(f"\n  WHY MANN-WHITNEY U TEST?")
    print(f"  The Mann-Whitney U Test was selected as the appropriate statistical")
    print(f"  method for this analysis because:")
    print(f"    • It is the non-parametric alternative to the Independent T-Test.")
    print(f"    • It does not assume that the outcome variable is normally distributed.")
    print(f"    • It is suitable when data is skewed, contains outliers, or when")
    print(f"      sample sizes are small.")
    print(f"    • It compares the rank-order distributions of two independent groups.")
    print(f"    • It tests whether one group tends to have systematically larger or")
    print(f"      smaller values than the other group.")

    print(f"\n  STATISTICAL FINDING:")
    if sig:
        print(f"  A statistically significant difference was detected in '{outcome_col}'")
        print(f"  between Group '{l1}' and Group '{l2}'")
        print(f"  (U = {results['u_stat']:.2f}, p = {p:.4f}, r = {r:.4f}).")
        print(f"\n  The median '{outcome_col}' was higher in Group '{higher_group}'")
        print(f"  (Median = {higher_med:,.4f}) compared to Group '{lower_group}'")
        print(f"  (Median = {lower_med:,.4f}).")
    else:
        print(f"  No statistically significant difference was detected in '{outcome_col}'")
        print(f"  between Group '{l1}' and Group '{l2}'")
        print(f"  (U = {results['u_stat']:.2f}, p = {p:.4f}, r = {r:.4f}).")
        print(f"\n  The median '{outcome_col}' for Group '{l1}'")
        print(f"  (Median = {med1:,.4f}) was not meaningfully different from")
        print(f"  Group '{l2}' (Median = {med2:,.4f}).")

    print(f"\n  EFFECT SIZE COMMENTARY:")
    abs_r = abs(r)
    effect_desc = results["effect_label"]
    print(f"  The rank-biserial correlation (r = {r:.4f}) indicates a {effect_desc} effect.")
    if abs_r < 0.10:
        print(f"  The practical difference between groups is negligible.")
    elif abs_r < 0.30:
        print(f"  There is a small practical difference between the groups.")
    elif abs_r < 0.50:
        print(f"  There is a moderate practical difference between the groups.")
    else:
        print(f"  There is a large and practically meaningful difference between groups.")

    print(f"\n  95% CONFIDENCE INTERVAL:")
    print(f"  The Hodges-Lehmann estimator of {results['hl_estimate']:.4f} with")
    print(f"  95% CI [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}] represents")
    print(f"  the estimated median difference (Group {l1} − Group {l2}).")
    if results["ci_lower"] > 0 or results["ci_upper"] < 0:
        print(f"  The confidence interval does not include zero, consistent with")
        print(f"  a statistically significant difference.")
    else:
        print(f"  The confidence interval includes zero, consistent with")
        print(f"  no statistically significant difference.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance alone does not establish clinical significance.")
    print(f"  Clinicians should consider the magnitude of difference (effect size),")
    print(f"  the clinical context, sample representativeness, and patient-specific")
    print(f"  factors when interpreting these results.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Mann-Whitney U Test assumes independence of observations.")
    print(f"  • Results are sensitive to ties in rank values.")
    print(f"  • This is an observational analysis and does not establish causality.")
    print(f"  • Confounding variables have not been adjusted for.")
    if results["n1"] < 20 or results["n2"] < 20:
        print(f"  • One or both groups have fewer than 20 observations.")
        print(f"    Interpret results with caution due to small sample size.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 normality: dict,
                 outcome_col: str,
                 group_col: str,
                 dataset_path: str) -> str:
    """
    Build the full text report as a string.
    Mirrors the printed output in a structured format.
    """
    l1, l2 = results["label1"], results["label2"]
    p = results["p_value"]
    r = results["r"]
    sig = p < 0.05
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Mann-Whitney U Test")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append(f"  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    lines.append(f"  Outcome Variable  : {outcome_col}")
    lines.append(f"  Grouping Variable : {group_col}")
    lines.append("")
    hdr = f"  {'Statistic':<28} {str(l1):>18} {str(l2):>18}"
    lines.append(hdr)
    lines.append(f"  {'-'*28} {'-'*18} {'-'*18}")

    def fmt(val):
        if isinstance(val, float) and math.isnan(val):
            return "N/A"
        return f"{val:,.4f}" if isinstance(val, float) else str(val)

    stat_rows = [
        ("Sample Size (n)",    "n"),
        ("Mean",               "mean"),
        ("Median",             "median"),
        ("Std Deviation",      "std"),
        ("Std Error of Mean",  "sem"),
        ("Minimum",            "min"),
        ("Maximum",            "max"),
        ("25th Percentile",    "q1"),
        ("75th Percentile",    "q3"),
        ("IQR (Q3 - Q1)",      "iqr"),
        ("Skewness",           "skewness"),
        ("Kurtosis",           "kurtosis"),
    ]
    for name, key in stat_rows:
        v0 = fmt(desc[l1][key])
        v1 = fmt(desc[l2][key])
        lines.append(f"  {name:<28} {v0:>18} {v1:>18}")

    # Assumption Checks
    lines.append("")
    lines.append(f"  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append(f"  [1] Independence of Observations: ASSUMED")
    lines.append(f"  [2] Normality (Shapiro-Wilk Test):")
    for label, res in normality.items():
        if res["note"]:
            lines.append(f"       Group {label}: {res['note']}")
        else:
            verdict = "Normal" if res["normal"] else "Non-Normal"
            lines.append(f"       Group {label}: W={res['stat']:.4f}, p={res['p']:.4f} → {verdict}")
    lines.append(f"  [3] Mann-Whitney U Test does not require normally distributed data.")
    lines.append(f"  [4] No equal variance assumption required.")

    # Test Results
    lines.append("")
    lines.append(f"  MANN-WHITNEY U TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test               : Mann-Whitney U Test (Two-Tailed)")
    lines.append(f"  Null Hypothesis    : Distributions equal across both groups")
    lines.append(f"  Significance Level : α = 0.05")
    lines.append(f"  U Statistic ({l1})  : {results['u_stat']:.4f}")
    lines.append(f"  U Statistic ({l2})  : {results['u_stat_g2']:.4f}")
    lines.append(f"  Z Score (approx.)  : {results['z_score']:.4f}")
    lines.append(f"  p-value            : {p:.4f}")
    lines.append(f"  Median ({l1})       : {desc[l1]['median']:,.4f}")
    lines.append(f"  Median ({l2})       : {desc[l2]['median']:,.4f}")
    lines.append(f"  Sample Size ({l1})  : {results['n1']:,}")
    lines.append(f"  Sample Size ({l2})  : {results['n2']:,}")
    lines.append(f"  Total Sample Size  : {results['total_n']:,}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result             : {sig_str}")

    # Effect Size
    lines.append("")
    lines.append(f"  EFFECT SIZE")
    lines.append(SEPARATOR2)
    lines.append(f"  Rank-Biserial Correlation (r)  : {r:.4f}")
    lines.append(f"  Effect Size Interpretation     : {results['effect_label']}")
    lines.append(f"  Hodges-Lehmann Estimator       : {results['hl_estimate']:.4f}")
    lines.append(f"  95% CI                         : [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    # Interpretation
    lines.append("")
    lines.append(f"  INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append(f"  Statistically significant difference detected between groups.")
        lines.append(f"  (U = {results['u_stat']:.2f}, p = {p:.4f}, r = {r:.4f})")
    else:
        lines.append(f"  No statistically significant difference detected between groups.")
        lines.append(f"  (U = {results['u_stat']:.2f}, p = {p:.4f}, r = {r:.4f})")
    lines.append(f"")
    lines.append(f"  The Mann-Whitney U Test was used as the non-parametric alternative")
    lines.append(f"  to the Independent T-Test. It is appropriate when data is not")
    lines.append(f"  normally distributed and compares rank-based distributions.")
    lines.append(f"  Statistical significance does not imply clinical significance.")
    lines.append(f"  Effect size (r = {r:.4f}) is classified as: {results['effect_label']}.")

    lines.append("")
    lines.append(SEPARATOR)
    lines.append("  END OF REPORT")
    lines.append(SEPARATOR)
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str):
    """Save the report to the outputs directory."""
    print_section("STEP 7 | SAVE REPORT")
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

def run_mann_whitney_u_test():
    """
    Main entry point for the Mann-Whitney U Test module.
    Orchestrates the full workflow from data loading to report generation.
    """
    print_header()
    print("  This module performs the Mann-Whitney U Test, a non-parametric")
    print("  method for comparing two independent groups.")
    print("  Suitable for continuous or ordinal outcome variables.")

    # Step 1: Load CSV
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select outcome column
    outcome_col = select_outcome_column(df)
    if outcome_col is None:
        sys.exit(1)

    # Step 4: Select grouping column
    group_col = select_grouping_column(df, outcome_col)
    if group_col is None:
        sys.exit(1)

    # Step 5: Clean data
    clean_df = clean_data(df, outcome_col, group_col)
    if clean_df is None:
        sys.exit(1)

    # Step 6: Extract groups
    groups = clean_df[group_col].unique()
    label1, label2 = groups[0], groups[1]

    group1 = clean_df.loc[clean_df[group_col] == label1, outcome_col].to_numpy(dtype=float)
    group2 = clean_df.loc[clean_df[group_col] == label2, outcome_col].to_numpy(dtype=float)

    group_data = {label1: group1, label2: group2}

    print(f"\n  Group Summary After Cleaning:")
    print(f"    Group '{label1}': n = {len(group1):,}")
    print(f"    Group '{label2}': n = {len(group2):,}")

    # Descriptive stats
    desc = compute_descriptive_stats(group_data)
    display_descriptive_stats(desc, outcome_col, group_col)

    # Normality checks
    normality = check_normality(group_data)
    display_assumption_checks(normality, group_data)

    # Run test
    print_section("STEP 6 | RUNNING MANN-WHITNEY U TEST")
    results = run_mann_whitney(group1, group2, label1, label2)
    if results is None:
        print("\n  [FATAL] Mann-Whitney U Test could not be completed.")
        sys.exit(1)

    display_test_results(results, desc, outcome_col)
    display_effect_size(results)
    display_interpretation(results, desc, normality, outcome_col, group_col)

    # Build and save report
    report_text = build_report(results, desc, normality, outcome_col, group_col, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Mann-Whitney U Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_mann_whitney_u_test()