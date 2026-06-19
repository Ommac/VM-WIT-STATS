
import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, shapiro

warnings.filterwarnings("ignore")


SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "kruskal_wallis_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Kruskal-Wallis Test
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_effectively_constant(series: pd.Series, tol: float = 1e-10) -> bool:
    return bool((series.max() - series.min()) < tol)


def eta_squared_h(h_stat: float, n_total: int, n_groups: int) -> float:
    """
    Compute Eta Squared (η²H) as effect size for Kruskal-Wallis Test.

        η²H = (H - k + 1) / (N - k)

    where k = number of groups, N = total observations.
    Clamped to 0.0 to prevent negative values caused by small H statistics.
    Returns NaN when the denominator is zero or undefined.
    """
    denominator = n_total - n_groups
    if denominator <= 0:
        return float("nan")
    return max(0.0, (h_stat - n_groups + 1) / denominator)


def interpret_eta_squared(eta2: float) -> str:
    """
    Interpret Eta Squared using standard benchmarks:
      η² < 0.01 → Negligible
      η² < 0.06 → Small
      η² < 0.14 → Medium
      η² ≥ 0.14 → Large
    """
    if math.isnan(eta2):
        return "N/A"
    if eta2 < 0.01:
        return "Negligible"
    elif eta2 < 0.06:
        return "Small"
    elif eta2 < 0.14:
        return "Medium"
    else:
        return "Large"


def fmt_val(val) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, float):
        return f"{val:,.4f}"
    return str(val)


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

def select_outcome_column(df: pd.DataFrame) -> str | None:
    print_section("STEP 3 | SELECT OUTCOME VARIABLE")
    print("\n  The outcome variable must be a continuous numeric column.")
    print("  Example: charges, meanbp, age, temperature, los, etc.")

    while True:
        col = input("\n  Enter outcome column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue

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
    print_section("STEP 4 | SELECT GROUPING VARIABLE")
    print("\n  The grouping variable must have 3 or more unique groups.")
    print("  Example: dzclass, race, edu, income, etc.")
    print("  Note: For exactly 2 groups, use the Mann-Whitney U Test module.")

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

        valid_groups = df[col].dropna().unique()
        n_groups = len(valid_groups)

        if n_groups < 3:
            print(f"  [ERROR] Column '{col}' has only {n_groups} unique group(s): {list(valid_groups)}")
            print("          Kruskal-Wallis Test requires 3 or more groups.")
            if n_groups == 2:
                print("          For 2 groups, use the Mann-Whitney U Test module.")
            continue

        print(f"  [OK] Grouping column selected: '{col}'")
        print(f"       Groups found ({n_groups}): {list(valid_groups)}")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               outcome_col: str,
               group_col: str) -> pd.DataFrame | None:
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    working = df[[outcome_col, group_col]].copy()
    original_rows = len(working)
    print(f"\n  Original rows : {original_rows:,}")

    working[outcome_col] = pd.to_numeric(working[outcome_col], errors="coerce")

    before = len(working)
    working = working.dropna(subset=[group_col])
    dropped_group_nan = before - len(working)
    if dropped_group_nan > 0:
        print(f"  Dropped (NaN in group column)    : {dropped_group_nan:,} row(s)")

    before = len(working)
    working = working.dropna(subset=[outcome_col])
    dropped_outcome_nan = before - len(working)
    if dropped_outcome_nan > 0:
        print(f"  Dropped (NaN in outcome column)  : {dropped_outcome_nan:,} row(s)")

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

    unique_groups = working[group_col].unique()
    n_groups = len(unique_groups)

    if n_groups < 3:
        print(f"\n  [ERROR] After cleaning, only {n_groups} group(s) remain: {list(unique_groups)}")
        print("          Kruskal-Wallis Test requires at least 3 groups.")
        return None

    # Check for empty or constant groups
    warnings_issued = False
    for grp in unique_groups:
        grp_data = working.loc[working[group_col] == grp, outcome_col]
        if len(grp_data) == 0:
            print(f"  [WARNING] Group '{grp}' is empty after cleaning.")
            warnings_issued = True
        elif len(grp_data) < 2:
            print(f"  [WARNING] Group '{grp}' has only {len(grp_data)} observation(s). Results may be unreliable.")
            warnings_issued = True
        elif is_effectively_constant(grp_data):
            print(f"  [WARNING] Group '{grp}' has zero variance (all values identical).")
            warnings_issued = True

    if not warnings_issued:
        print(f"\n  [OK] Data is clean and ready for analysis.")
    else:
        print(f"\n  [OK] Data cleaned. Proceed with caution due to warnings above.")

    return working


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(group_data: dict) -> dict:
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
    print_section("DESCRIPTIVE STATISTICS")
    labels = list(desc.keys())

    print(f"\n  Outcome Variable  : {outcome_col}")
    print(f"  Grouping Variable : {group_col}")
    print(f"  Number of Groups  : {len(labels)}\n")

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

    col_width = 18
    label_width = 28

    # Header
    header = f"  {'Statistic':<{label_width}}"
    for lbl in labels:
        header += f" {str(lbl)[:col_width]:>{col_width}}"
    print(header)

    divider = f"  {'-'*label_width}"
    for _ in labels:
        divider += f" {'-'*col_width}"
    print(divider)

    for display_name, key in stat_rows:
        row = f"  {display_name:<{label_width}}"
        for lbl in labels:
            row += f" {fmt_val(desc[lbl][key]):>{col_width}}"
        print(row)


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def check_normality(group_data: dict) -> dict:
    results = {}
    for label, values in group_data.items():
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n < 3:
            results[label] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": None, "note": "Too few samples (n < 3)"
            }
        elif n > 5000:
            results[label] = {
                "stat": float("nan"), "p": float("nan"),
                "normal": None,
                "note": "n > 5000: Shapiro-Wilk not computed (large sample)"
            }
        else:
            stat, p = shapiro(arr)
            results[label] = {
                "stat": float(stat), "p": float(p),
                "normal": p > 0.05,
                "note": ""
            }
    return results


def display_assumption_checks(normality: dict, group_data: dict):
    print_section("ASSUMPTION CHECKS")

    print("\n  [1] Independence of Observations")
    print("      ASSUMED: Each observation belongs to exactly one group.")
    print("      Ensure data collection followed independent sampling design.")

    print("\n  [2] Normality of Outcome Variable (Shapiro-Wilk Test)")
    print(f"      H0: Data is normally distributed  (p > 0.05 = Normal)")
    print(f"      H1: Data is NOT normally distributed (p ≤ 0.05 = Non-Normal)\n")

    for label, res in normality.items():
        n = len(group_data[label])
        print(f"      Group: {label}  (n = {n:,})")
        if res["note"]:
            print(f"        Note   : {res['note']}")
        else:
            verdict = "Normal" if res["normal"] is True else "Non-Normal"
            print(f"        W Stat : {res['stat']:.4f}")
            print(f"        p-value: {res['p']:.4f}")
            print(f"        Result : {verdict}")
        print()

    print("  [3] Scale of Measurement")
    print("      ASSUMED: Outcome variable is at least ordinal scale.")
    print("      Continuous and ordinal data are both appropriate.")

    print("\n  [4] Why Kruskal-Wallis Test?")
    print("      The Kruskal-Wallis Test was selected because:")
    print("      - It is the non-parametric alternative to One-Way ANOVA.")
    print("      - It does not assume normally distributed data.")
    print("      - It compares rank-based distributions across 3+ groups.")
    print("      - It is robust to outliers and skewed distributions.")
    print("      - It tests whether at least one group tends to differ")
    print("        systematically from the others.")
    print("      - No assumption of equal variances (homoscedasticity) required.")


# ======================================================================
# KRUSKAL-WALLIS TEST
# ======================================================================

def run_kruskal_wallis(group_data: dict) -> dict | None:
    labels = list(group_data.keys())
    arrays = [np.array(group_data[lbl], dtype=float) for lbl in labels]
    n_groups = len(arrays)
    n_total = sum(len(a) for a in arrays)

    # Guard: minimum sample sizes
    for lbl, arr in zip(labels, arrays):
        if len(arr) < 1:
            print(f"  [ERROR] Group '{lbl}' is empty. Cannot run Kruskal-Wallis Test.")
            return None
        if len(arr) < 2:
            print(f"  [WARNING] Group '{lbl}' has n={len(arr)}. Results may be unreliable.")

    try:
        h_stat, p_value = kruskal(*arrays)
    except ValueError as exc:
        print(f"  [ERROR] Kruskal-Wallis Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Kruskal-Wallis Test.\n          Details: {exc}")
        return None

    if math.isnan(h_stat) or math.isnan(p_value):
        print("  [ERROR] Kruskal-Wallis Test returned NaN results.")
        print("          This may indicate insufficient data or constant values.")
        return None

    df_kw = n_groups - 1
    eta2 = eta_squared_h(h_stat, n_total, n_groups)
    effect_label = interpret_eta_squared(eta2)

    return {
        "h_stat"       : float(h_stat),
        "p_value"      : float(p_value),
        "df"           : df_kw,
        "n_total"      : n_total,
        "n_groups"     : n_groups,
        "labels"       : labels,
        "eta2"         : float(eta2),
        "effect_label" : effect_label,
    }


# ======================================================================
# POST-HOC ANALYSIS
# ======================================================================

def run_posthoc(group_data: dict) -> list:
    """
    Pairwise Mann-Whitney U comparisons with Bonferroni correction.
    Returns list of dicts, one per pair.
    """
    labels = list(group_data.keys())
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j]))

    n_comparisons = len(pairs)
    results = []

    for lbl1, lbl2 in pairs:
        arr1 = np.array(group_data[lbl1], dtype=float)
        arr2 = np.array(group_data[lbl2], dtype=float)
        try:
            u_stat, p_raw = mannwhitneyu(arr1, arr2, alternative="two-sided")
            p_bonf = min(float(p_raw) * n_comparisons, 1.0)
            results.append({
                "group1"    : lbl1,
                "group2"    : lbl2,
                "u_stat"    : float(u_stat),
                "p_raw"     : float(p_raw),
                "p_bonf"    : float(p_bonf),
                "n1"        : len(arr1),
                "n2"        : len(arr2),
                "sig_raw"   : float(p_raw) < 0.05,
                "sig_bonf"  : float(p_bonf) < 0.05,
                "error"     : None,
            })
        except Exception as exc:
            results.append({
                "group1"    : lbl1,
                "group2"    : lbl2,
                "u_stat"    : float("nan"),
                "p_raw"     : float("nan"),
                "p_bonf"    : float("nan"),
                "n1"        : len(arr1),
                "n2"        : len(arr2),
                "sig_raw"   : False,
                "sig_bonf"  : False,
                "error"     : str(exc),
            })

    return results


def display_posthoc(posthoc_results: list, n_comparisons: int):
    print_section("POST-HOC ANALYSIS (Pairwise Mann-Whitney U + Bonferroni Correction)")

    print(f"\n  Total pairwise comparisons : {n_comparisons}")
    print(f"  Correction method          : Bonferroni")
    print(f"  Corrected α threshold      : {0.05 / n_comparisons:.4f}")
    print(f"  (α = 0.05 / {n_comparisons} comparisons)\n")

    col_g = 20
    col_u = 14
    col_p = 12
    col_pb = 16
    col_s = 16

    hdr = (f"  {'Group 1':<{col_g}} {'Group 2':<{col_g}}"
           f" {'U Statistic':>{col_u}} {'p-value':>{col_p}}"
           f" {'Bonf. p-value':>{col_pb}} {'Significant':>{col_s}}")
    print(hdr)
    print(f"  {'-'*col_g} {'-'*col_g} {'-'*col_u} {'-'*col_p} {'-'*col_pb} {'-'*col_s}")

    for res in posthoc_results:
        if res["error"]:
            row = (f"  {str(res['group1']):<{col_g}} {str(res['group2']):<{col_g}}"
                   f" {'ERROR':>{col_u}} {'N/A':>{col_p}}"
                   f" {'N/A':>{col_pb}} {'N/A':>{col_s}}")
        else:
            sig_str = "YES *" if res["sig_bonf"] else "no"
            row = (f"  {str(res['group1']):<{col_g}} {str(res['group2']):<{col_g}}"
                   f" {res['u_stat']:>{col_u},.2f}"
                   f" {res['p_raw']:>{col_p}.4f}"
                   f" {res['p_bonf']:>{col_pb}.4f}"
                   f" {sig_str:>{col_s}}")
        print(row)

    sig_pairs = [r for r in posthoc_results if r["sig_bonf"] and not r["error"]]
    print(f"\n  Significant pairs after Bonferroni correction: {len(sig_pairs)} / {n_comparisons}")
    if sig_pairs:
        print(f"\n  Significant pairs:")
        for r in sig_pairs:
            print(f"    {r['group1']} vs {r['group2']}  "
                  f"(U = {r['u_stat']:.2f}, p_bonf = {r['p_bonf']:.4f})")
    else:
        print("  No pairs reached significance after Bonferroni correction.")


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, desc: dict, outcome_col: str):
    print_section("KRUSKAL-WALLIS TEST RESULTS")

    sig = results["p_value"] < 0.05
    labels = results["labels"]

    print(f"\n  Test             : Kruskal-Wallis H Test")
    print(f"  Null Hypothesis  : All groups have identical distributions")
    print(f"                     for '{outcome_col}'.")
    print(f"  Alt. Hypothesis  : At least one group distribution differs.")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'H Statistic':<40} {results['h_stat']:>20.4f}")
    print(f"  {'Degrees of Freedom':<40} {results['df']:>20}")
    print(f"  {'p-value':<40} {results['p_value']:>20.4f}")
    print(f"  {'Total Sample Size':<40} {results['n_total']:>20,}")
    print(f"  {'Number of Groups':<40} {results['n_groups']:>20}")
    print(f"  {SEPARATOR2[:68]}")

    print(f"\n  Group Medians and Means:")
    print(f"  {'Group':<30} {'n':>8} {'Median':>14} {'Mean':>14}")
    print(f"  {'-'*30} {'-'*8} {'-'*14} {'-'*14}")
    for lbl in labels:
        d = desc[lbl]
        print(f"  {str(lbl):<30} {d['n']:>8,} {d['median']:>14,.4f} {d['mean']:>14,.4f}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result  : {sig_str}")
    print(f"  p-value : {results['p_value']:.4f}  (α = 0.05)")


def display_effect_size(results: dict):
    print_section("EFFECT SIZE")

    eta2 = results["eta2"]
    print(f"\n  Effect Size Measure : Eta Squared (η²H)")
    print(f"  Formula             : η²H = (H - k + 1) / (N - k)\n")

    print(f"  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'H Statistic':<40} {results['h_stat']:>20.4f}")
    print(f"  {'Total N':<40} {results['n_total']:>20,}")
    print(f"  {'Eta Squared (η²H)':<40} {eta2:>20.4f}")
    print(f"  {'Effect Size Interpretation':<40} {results['effect_label']:>20}")

    print(f"\n  Effect Size Benchmarks (Cohen 1992 / Tomczak & Tomczak 2014):")
    print(f"    η² < 0.01 → Negligible")
    print(f"    η² < 0.06 → Small")
    print(f"    η² < 0.14 → Medium")
    print(f"    η² ≥ 0.14 → Large")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            normality: dict,
                            posthoc: list,
                            outcome_col: str,
                            group_col: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    labels = results["labels"]
    p = results["p_value"]
    eta2 = results["eta2"]
    h = results["h_stat"]
    sig = p < 0.05

    print(f"\n  OUTCOME VARIABLE  : {outcome_col}")
    print(f"  GROUPING VARIABLE : {group_col}")
    print(f"  NUMBER OF GROUPS  : {results['n_groups']}")
    print(f"  TOTAL SAMPLE SIZE : {results['n_total']:,}")

    print(f"\n  WHY KRUSKAL-WALLIS TEST?")
    print(f"  The Kruskal-Wallis Test was selected as the appropriate statistical")
    print(f"  method for this analysis because:")
    print(f"    • It is the non-parametric alternative to One-Way ANOVA.")
    print(f"    • It does not assume normally distributed data.")
    print(f"    • It is suitable when data is skewed, contains outliers, or when")
    print(f"      the normality assumption of ANOVA is violated.")
    print(f"    • It compares rank-order distributions across {results['n_groups']} groups.")
    print(f"    • It tests whether at least one group tends to have systematically")
    print(f"      different values compared to the other groups.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        print(f"  At least one group differs significantly from the others.")
        print(f"  (H({results['df']}) = {h:.4f}, p = {p:.4f}, η²H = {eta2:.4f})")
        print(f"\n  The Kruskal-Wallis Test revealed a statistically significant")
        print(f"  difference in '{outcome_col}' across the {results['n_groups']} groups")
        print(f"  of '{group_col}'.")

        # Identify highest and lowest median groups
        medians = {lbl: desc[lbl]["median"] for lbl in labels}
        highest_grp = max(medians, key=medians.get)
        lowest_grp  = min(medians, key=medians.get)
        print(f"\n  Group with highest median: '{highest_grp}' "
              f"(Median = {medians[highest_grp]:,.4f})")
        print(f"  Group with lowest  median: '{lowest_grp}' "
              f"(Median = {medians[lowest_grp]:,.4f})")
    else:
        print(f"  No statistically significant difference was detected among the groups.")
        print(f"  (H({results['df']}) = {h:.4f}, p = {p:.4f}, η²H = {eta2:.4f})")
        print(f"\n  The Kruskal-Wallis Test found no significant difference in")
        print(f"  '{outcome_col}' across the {results['n_groups']} groups of '{group_col}'.")
        print(f"  The observed differences in group medians may be due to chance.")

    print(f"\n  EFFECT SIZE COMMENTARY:")
    abs_eta = abs(eta2)
    effect_desc = results["effect_label"]
    print(f"  Eta Squared (η²H = {eta2:.4f}) indicates a {effect_desc} effect.")
    if abs_eta < 0.01:
        print(f"  The proportion of variance explained by group membership is negligible.")
    elif abs_eta < 0.06:
        print(f"  Group membership explains a small proportion of the variance in outcome.")
    elif abs_eta < 0.14:
        print(f"  Group membership explains a moderate proportion of the outcome variance.")
    else:
        print(f"  Group membership explains a large proportion of the outcome variance.")
        print(f"  This is a clinically meaningful finding.")

    if sig and posthoc:
        sig_pairs = [r for r in posthoc if r["sig_bonf"] and not r["error"]]
        print(f"\n  POST-HOC COMMENTARY:")
        if sig_pairs:
            print(f"  After Bonferroni correction, {len(sig_pairs)} pair(s) reached significance:")
            for r in sig_pairs:
                print(f"    • '{r['group1']}' vs '{r['group2']}'"
                      f"  (U = {r['u_stat']:.2f}, p_bonf = {r['p_bonf']:.4f})")
        else:
            print(f"  Although the overall test is significant, no individual pairwise")
            print(f"  comparison survived Bonferroni correction. This may indicate")
            print(f"  a diffuse pattern of differences across multiple groups.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance alone does not establish clinical significance.")
    print(f"  Clinicians should consider the magnitude of difference (effect size),")
    print(f"  the clinical context, sample representativeness, and patient-specific")
    print(f"  factors when interpreting these results.")
    print(f"  Rank-based methods compare distributional tendencies, not absolute values.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Kruskal-Wallis Test assumes independence of all observations.")
    print(f"  • Results are sensitive to tied ranks in the data.")
    print(f"  • This is an observational analysis and does not establish causality.")
    print(f"  • Confounding variables have not been adjusted for in this analysis.")
    print(f"  • The Bonferroni correction is conservative; some true differences")
    print(f"    may not survive correction when there are many comparisons.")
    small_groups = [lbl for lbl in labels if desc[lbl]["n"] < 20]
    if small_groups:
        print(f"  • Small sample size(s) detected in group(s): {small_groups}")
        print(f"    Interpret results with caution due to limited statistical power.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 normality: dict,
                 posthoc: list,
                 outcome_col: str,
                 group_col: str,
                 dataset_path: str) -> str:
    labels = results["labels"]
    p = results["p_value"]
    h = results["h_stat"]
    eta2 = results["eta2"]
    sig = p < 0.05
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Kruskal-Wallis Test")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    lines.append(f"  Outcome Variable  : {outcome_col}")
    lines.append(f"  Grouping Variable : {group_col}")
    lines.append(f"  Number of Groups  : {len(labels)}")
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
    col_w = 16
    lbl_w = 28
    hdr = f"  {'Statistic':<{lbl_w}}"
    for lbl in labels:
        hdr += f" {str(lbl)[:col_w]:>{col_w}}"
    lines.append(hdr)
    div = f"  {'-'*lbl_w}"
    for _ in labels:
        div += f" {'-'*col_w}"
    lines.append(div)
    for name, key in stat_rows:
        row = f"  {name:<{lbl_w}}"
        for lbl in labels:
            row += f" {fmt_val(desc[lbl][key]):>{col_w}}"
        lines.append(row)

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append("  [1] Independence of Observations: ASSUMED")
    lines.append("  [2] Normality (Shapiro-Wilk Test):")
    for label, res in normality.items():
        if res["note"]:
            lines.append(f"       Group {label}: {res['note']}")
        else:
            verdict = "Normal" if res["normal"] is True else "Non-Normal"
            lines.append(f"       Group {label}: W={res['stat']:.4f}, p={res['p']:.4f} → {verdict}")
    lines.append("  [3] Kruskal-Wallis does not require normally distributed data.")
    lines.append("  [4] No equal variance assumption required.")
    lines.append("  [5] Non-parametric alternative to One-Way ANOVA.")

    # Test Results
    lines.append("")
    lines.append("  KRUSKAL-WALLIS TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test               : Kruskal-Wallis H Test")
    lines.append(f"  H Statistic        : {h:.4f}")
    lines.append(f"  Degrees of Freedom : {results['df']}")
    lines.append(f"  p-value            : {p:.4f}")
    lines.append(f"  Total N            : {results['n_total']:,}")
    lines.append(f"  Number of Groups   : {results['n_groups']}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result             : {sig_str}")
    lines.append("")
    lines.append(f"  {'Group':<30} {'n':>8} {'Median':>14} {'Mean':>14}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*14} {'-'*14}")
    for lbl in labels:
        d = desc[lbl]
        lines.append(f"  {str(lbl):<30} {d['n']:>8,} {d['median']:>14,.4f} {d['mean']:>14,.4f}")

    # Post-Hoc Analysis
    lines.append("")
    lines.append("  POST-HOC ANALYSIS (Pairwise Mann-Whitney U + Bonferroni Correction)")
    lines.append(SEPARATOR2)
    if not posthoc:
        lines.append("  Post-hoc analysis was not performed.")
        lines.append("  Overall Kruskal-Wallis p ≥ 0.05; pairwise comparisons are not warranted.")
    else:
        n_comp = len(posthoc)
        lines.append(f"  Total comparisons      : {n_comp}")
        lines.append(f"  Correction             : Bonferroni")
        bonf_alpha = 0.05 / n_comp if n_comp > 0 else 0.05
        lines.append(f"  Corrected α            : {bonf_alpha:.4f}")
        lines.append("")
        col_g = 20
        col_u = 14
        col_p = 12
        col_pb = 16
        col_s = 14
        ph_hdr = (f"  {'Group 1':<{col_g}} {'Group 2':<{col_g}}"
                  f" {'U Statistic':>{col_u}} {'p-value':>{col_p}}"
                  f" {'Bonf. p-value':>{col_pb}} {'Significant':>{col_s}}")
        lines.append(ph_hdr)
        lines.append(f"  {'-'*col_g} {'-'*col_g} {'-'*col_u} {'-'*col_p} {'-'*col_pb} {'-'*col_s}")
        for res in posthoc:
            if res["error"]:
                row = (f"  {str(res['group1']):<{col_g}} {str(res['group2']):<{col_g}}"
                       f" {'ERROR':>{col_u}} {'N/A':>{col_p}} {'N/A':>{col_pb}} {'N/A':>{col_s}}")
            else:
                sig_str_ph = "YES *" if res["sig_bonf"] else "no"
                row = (f"  {str(res['group1']):<{col_g}} {str(res['group2']):<{col_g}}"
                       f" {res['u_stat']:>{col_u},.2f}"
                       f" {res['p_raw']:>{col_p}.4f}"
                       f" {res['p_bonf']:>{col_pb}.4f}"
                       f" {sig_str_ph:>{col_s}}")
            lines.append(row)

    # Effect Size
    lines.append("")
    lines.append("  EFFECT SIZE")
    lines.append(SEPARATOR2)
    lines.append(f"  Eta Squared (η²H)              : {eta2:.4f}")
    lines.append(f"  Effect Size Interpretation     : {results['effect_label']}")
    lines.append(f"  Formula                        : η²H = (H - k + 1) / (N - k)")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append("  At least one group differs significantly from the others.")
        lines.append(f"  H({results['df']}) = {h:.4f}, p = {p:.4f}, η²H = {eta2:.4f}")
    else:
        lines.append("  No statistically significant difference was detected among the groups.")
        lines.append(f"  H({results['df']}) = {h:.4f}, p = {p:.4f}, η²H = {eta2:.4f}")
    lines.append("")
    lines.append("  The Kruskal-Wallis Test was used as the non-parametric alternative")
    lines.append("  to One-Way ANOVA. It does not require normally distributed data and")
    lines.append("  compares rank distributions across 3 or more independent groups.")
    lines.append(f"  Effect size (η²H = {eta2:.4f}) is classified as: {results['effect_label']}.")
    lines.append("  Statistical significance does not imply clinical significance.")
    lines.append("  Confounding variables have not been adjusted for in this analysis.")

    lines.append("")
    lines.append(SEPARATOR)
    lines.append("  END OF REPORT")
    lines.append(SEPARATOR)
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str):
    print_section("STEP 8 | SAVE REPORT")
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

def run_kruskal_wallis_test():
    print_header()
    print("  This module performs the Kruskal-Wallis H Test, a non-parametric")
    print("  method for comparing three or more independent groups.")
    print("  It is the non-parametric alternative to One-Way ANOVA.")

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

    # Step 6: Build group data dict
    group_labels = sorted(clean_df[group_col].unique(), key=str)
    group_data = {}
    for lbl in group_labels:
        group_data[lbl] = clean_df.loc[
            clean_df[group_col] == lbl, outcome_col
        ].to_numpy(dtype=float)

    print(f"\n  Group Summary After Cleaning:")
    print(f"  {'Group':<30} {'n':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for lbl, arr in group_data.items():
        print(f"  {str(lbl):<30} {len(arr):>10,}")

    # Descriptive stats
    desc = compute_descriptive_stats(group_data)
    display_descriptive_stats(desc, outcome_col, group_col)

    # Normality checks
    normality = check_normality(group_data)
    display_assumption_checks(normality, group_data)

    # Run Kruskal-Wallis
    print_section("STEP 6 | RUNNING KRUSKAL-WALLIS TEST")
    kw_results = run_kruskal_wallis(group_data)
    if kw_results is None:
        print("\n  [FATAL] Kruskal-Wallis Test could not be completed.")
        sys.exit(1)

    display_test_results(kw_results, desc, outcome_col)

    # Post-hoc analysis — only run if overall test is significant
    print_section("STEP 7 | POST-HOC ANALYSIS")
    if kw_results["p_value"] < 0.05:
        print(f"\n  p < 0.05 detected. Running pairwise post-hoc comparisons...")
        posthoc = run_posthoc(group_data)
        n_comparisons = len(posthoc)
        display_posthoc(posthoc, n_comparisons)
    else:
        print(f"\n  p ≥ 0.05. Post-hoc analysis skipped.")
        print(f"  (Overall test is not significant; pairwise comparisons")
        print(f"   are not warranted and would inflate Type I error.)")
        posthoc = []

    display_effect_size(kw_results)
    display_interpretation(kw_results, desc, normality, posthoc, outcome_col, group_col)

    # Build and save report (always includes full post-hoc table)
    report_text = build_report(
        kw_results, desc, normality, posthoc,
        outcome_col, group_col, path
    )
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Kruskal-Wallis Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_kruskal_wallis_test()