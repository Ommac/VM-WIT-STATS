# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Fisher Exact Test
# File        : hypothesis_tests/fisher_exact_test.py
# Description : Exact test of association between two binary categorical
#               variables in a 2×2 contingency table.
#               Preferred over Chi-Square when expected frequencies < 5
#               or when sample sizes are small.
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "fisher_exact_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Fisher Exact Test
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


def odds_ratio_ci(a: int, b: int, c: int, d: int,
                  confidence: float = 0.95) -> tuple:
    """
    Calculate 95% confidence interval for Odds Ratio using the
    Woolf (logit) method:
        log(OR) ± Z × SE,  where SE = sqrt(1/a + 1/b + 1/c + 1/d)

    Returns (or_value, ci_lower, ci_upper).
    Handles zero cells via continuity correction (+0.5).
    Returns (NaN, NaN, NaN) if calculation is impossible.
    """
    # Apply continuity correction when any cell is zero
    a0, b0, c0, d0 = float(a), float(b), float(c), float(d)
    if 0 in (a, b, c, d):
        a0 += 0.5
        b0 += 0.5
        c0 += 0.5
        d0 += 0.5

    if b0 == 0 or c0 == 0:
        return float("nan"), float("nan"), float("nan")

    or_val = (a0 * d0) / (b0 * c0)

    if or_val <= 0:
        return float("nan"), float("nan"), float("nan")

    log_or = math.log(or_val)
    se = math.sqrt(1 / a0 + 1 / b0 + 1 / c0 + 1 / d0)
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = math.exp(log_or - z * se)
    ci_upper = math.exp(log_or + z * se)

    return or_val, ci_lower, ci_upper


def interpret_odds_ratio(or_val: float) -> tuple:
    """
    Interpret magnitude and direction of Odds Ratio.
    Returns (magnitude_label, direction_label).
    """
    if math.isnan(or_val) or math.isinf(or_val):
        return "N/A", "N/A"

    if or_val == 1.0:
        direction = "No association"
        magnitude = "None"
    elif or_val > 1.0:
        direction = "Positive association (increased odds in group 1)"
        if or_val <= 2.0:
            magnitude = "Weak"
        elif or_val <= 5.0:
            magnitude = "Moderate"
        else:
            magnitude = "Strong"
    else:
        direction = "Negative association (decreased odds in group 1)"
        inv = 1.0 / or_val if or_val > 0 else float("inf")
        if inv <= 2.0:
            magnitude = "Weak"
        elif inv <= 5.0:
            magnitude = "Moderate"
        else:
            magnitude = "Strong"

    return magnitude, direction


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

def select_binary_column(df: pd.DataFrame,
                          label: str,
                          exclude_col: str | None = None) -> str | None:
    """
    Prompt user to select a binary categorical column (exactly 2 categories).
    Validates existence, uniqueness, and binary constraint.
    """
    print(f"\n  The {label} must be a binary categorical column (exactly 2 categories).")
    print(f"  Example: sex (Male/Female), death (0/1), treatment (Drug/Placebo), etc.")

    while True:
        col = input(f"\n  Enter {label} column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found in dataset.")
            print(f"          Available columns: {list(df.columns[:10])} ...")
            continue
        if exclude_col and col == exclude_col:
            print(f"  [ERROR] {label} cannot be the same as the already selected column ('{exclude_col}').")
            continue

        valid_values = df[col].dropna()
        if len(valid_values) == 0:
            print(f"  [ERROR] Column '{col}' contains no non-missing values.")
            continue

        unique_cats = valid_values.astype(str).str.strip().unique()
        unique_cats = [c for c in unique_cats if c != ""]
        n_unique = len(unique_cats)

        if n_unique < 2:
            print(f"  [ERROR] Column '{col}' has only {n_unique} unique category: {list(unique_cats)}")
            print("          Fisher Exact Test requires exactly 2 categories per variable.")
            continue
        if n_unique > 2:
            print(f"  [ERROR] Column '{col}' has {n_unique} unique categories: {list(unique_cats[:10])}")
            print("          Fisher Exact Test requires exactly 2 categories per variable.")
            print("          For variables with more categories, use the Chi-Square Test module.")
            continue

        print(f"  [OK] {label} selected: '{col}'")
        print(f"       Categories (2): {list(unique_cats)}")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               var1: str,
               var2: str) -> pd.DataFrame | None:
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    working = df[[var1, var2]].copy()
    original_rows = len(working)
    print(f"\n  Original rows : {original_rows:,}")

    # Drop rows where either variable is NaN
    before = len(working)
    working = working.dropna(subset=[var1, var2])
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        print(f"  Dropped (NaN in either variable) : {dropped_nan:,} row(s)")

    # Coerce to string and strip
    working[var1] = working[var1].astype(str).str.strip()
    working[var2] = working[var2].astype(str).str.strip()

    # Remove empty strings
    before = len(working)
    working = working[(working[var1] != "") & (working[var2] != "")]
    dropped_empty = before - len(working)
    if dropped_empty > 0:
        print(f"  Dropped (empty string values)    : {dropped_empty:,} row(s)")

    clean_rows = len(working)
    total_dropped = original_rows - clean_rows
    print(f"\n  Total rows dropped : {total_dropped:,}")
    print(f"  Rows for analysis  : {clean_rows:,}")

    if clean_rows == 0:
        print("\n  [ERROR] No valid rows remain after cleaning.")
        return None

    # Re-check binary constraint after cleaning
    for var in [var1, var2]:
        cats = working[var].unique()
        if len(cats) < 2:
            print(f"\n  [ERROR] After cleaning, '{var}' has only {len(cats)} category: {list(cats)}")
            print("          Fisher Exact Test requires exactly 2 categories.")
            return None
        if len(cats) > 2:
            print(f"\n  [ERROR] After cleaning, '{var}' has {len(cats)} categories: {list(cats)}")
            print("          Fisher Exact Test requires exactly 2 categories per variable.")
            print("          Consider using the Chi-Square Test module instead.")
            return None

    print(f"\n  [OK] Data is clean and ready for analysis.")
    return working


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(df: pd.DataFrame,
                               var1: str,
                               var2: str) -> dict:
    desc = {}
    for var in [var1, var2]:
        counts = df[var].value_counts().sort_index()
        total = int(counts.sum())
        desc[var] = {
            "categories"  : list(counts.index),
            "counts"      : list(counts.values),
            "proportions" : [round(c / total * 100, 2) for c in counts.values],
            "n_categories": len(counts),
            "total"       : total,
        }
    return desc


def display_descriptive_stats(desc: dict, var1: str, var2: str):
    print_section("DESCRIPTIVE STATISTICS")

    for var in [var1, var2]:
        d = desc[var]
        print(f"\n  Variable             : {var}")
        print(f"  Total Valid N        : {d['total']:,}")
        print(f"  Number of Categories : {d['n_categories']}")
        print(f"\n  {'Category':<30} {'Count':>10} {'Percent':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for cat, cnt, pct in zip(d["categories"], d["counts"], d["proportions"]):
            print(f"  {str(cat):<30} {cnt:>10,} {pct:>9.2f}%")


# ======================================================================
# CONTINGENCY TABLE
# ======================================================================

def build_contingency_table(df: pd.DataFrame,
                             var1: str,
                             var2: str) -> pd.DataFrame:
    table = pd.crosstab(df[var1], df[var2])
    return table


def display_contingency_table(table: pd.DataFrame, var1: str, var2: str):
    print_section("CONTINGENCY TABLE (Observed Frequencies)")

    col_labels = [str(c) for c in table.columns]
    row_labels = [str(r) for r in table.index]
    n_total = int(table.values.sum())

    col_w = max(14, max(len(c) for c in col_labels) + 2)
    row_label_w = max(22, max(len(r) for r in row_labels) + 4)

    print(f"\n  Row variable    : {var1}")
    print(f"  Column variable : {var2}")
    print(f"  Total N         : {n_total:,}")
    print(f"  Table Dimension : 2 × 2\n")

    # Observed counts
    hdr = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr += f" {c[:col_w]:>{col_w}}"
    hdr += f" {'Row Total':>{col_w}}"
    print(hdr)
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))

    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_total = int(row_vals.sum())
        row_str = f"  {str(row_lbl)[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            row_str += f" {int(val):>{col_w},}"
        row_str += f" {row_total:>{col_w},}"
        print(row_str)

    col_totals = table.sum(axis=0)
    total_str = f"  {'Col Total'[:row_label_w]:<{row_label_w}}"
    for val in col_totals:
        total_str += f" {int(val):>{col_w},}"
    total_str += f" {n_total:>{col_w},}"
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    print(total_str)

    # Row percentages
    print(f"\n  ROW PERCENTAGES (% within {var1}):\n")
    print(hdr.split("Row Total")[0].rstrip())
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_total = row_vals.sum()
        row_str = f"  {str(row_lbl)[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            pct = (val / row_total * 100) if row_total > 0 else 0.0
            pct_str = f"{pct:.1f}%"
            row_str += f" {pct_str:>{col_w}}"
        print(row_str)

    # Column percentages
    print(f"\n  COLUMN PERCENTAGES (% within {var2}):\n")
    print(hdr.split("Row Total")[0].rstrip())
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    col_totals_arr = table.sum(axis=0)
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_str = f"  {str(row_lbl)[:row_label_w]:<{row_label_w}}"
        for col_lbl, val in zip(col_labels, row_vals):
            col_total = col_totals_arr[col_lbl] if col_lbl in col_totals_arr else 0
            pct = (val / col_total * 100) if col_total > 0 else 0.0
            pct_str = f"{pct:.1f}%"
            row_str += f" {pct_str:>{col_w}}"
        print(row_str)


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def display_assumption_checks(table: pd.DataFrame, var1: str, var2: str):
    print_section("ASSUMPTION CHECKS")

    n_total = int(table.values.sum())

    print("\n  [1] Categorical Variables")
    print(f"      CONFIRMED: Both '{var1}' and '{var2}' are categorical.")
    print(f"      Each variable has exactly 2 categories (binary).")

    print("\n  [2] Binary Variables — Exactly 2 Categories Each")
    cats1 = list(table.index)
    cats2 = list(table.columns)
    print(f"      '{var1}' categories : {cats1}")
    print(f"      '{var2}' categories : {cats2}")
    print(f"      CONFIRMED: 2×2 contingency table verified.")

    print("\n  [3] Independence of Observations")
    print("      ASSUMED: Each observation is independent of all others.")
    print("      Fisher Exact Test is not appropriate for matched or paired data.")
    print("      Ensure data collection followed an independent sampling design.")

    print("\n  [4] Fisher Exact Test vs Chi-Square Test")
    print("      Fisher Exact Test is preferred over Chi-Square when:")
    print("      - Sample sizes are small (N < 20).")
    print("      - Expected cell frequencies fall below 5.")
    print("      - The contingency table is sparse.")
    print(f"      - Total N = {n_total:,}.")
    if n_total < 20:
        print(f"      [NOTE] Small sample (N = {n_total:,}). Fisher Exact Test is strongly recommended.")
    elif n_total < 40:
        print(f"      [NOTE] Moderate sample (N = {n_total:,}). Fisher Exact Test is appropriate.")
    else:
        print(f"      [NOTE] Adequate sample (N = {n_total:,}). Both Fisher and Chi-Square are valid.")
        print(f"             Fisher Exact Test used here for exact inference regardless of sample size.")

    print("\n  [5] No Minimum Expected Frequency Requirement")
    print("      ADVANTAGE: Unlike Chi-Square, Fisher Exact Test does not require")
    print("      expected cell frequencies to be ≥ 5. It is an exact method that")
    print("      computes the exact probability of the observed table and all tables")
    print("      more extreme, conditional on fixed marginal totals.")

    print("\n  [6] Zero Cells")
    zero_cells = int(np.sum(table.values == 0))
    if zero_cells > 0:
        print(f"      [NOTE] {zero_cells} cell(s) contain zero counts.")
        print(f"             Fisher Exact Test handles zero cells correctly.")
        print(f"             Odds Ratio CI will use continuity correction (+0.5).")
    else:
        print("      CONFIRMED: No zero cells in the contingency table.")


# ======================================================================
# FISHER EXACT TEST
# ======================================================================

def run_fisher_exact(table: pd.DataFrame) -> dict | None:
    """
    Run Fisher Exact Test using scipy.stats.fisher_exact().
    Computes two-sided p-value, Odds Ratio, and 95% CI via Woolf method.
    Returns a dictionary of all computed statistics, or None on failure.
    """
    observed = table.values
    n_rows, n_cols = observed.shape

    if n_rows != 2 or n_cols != 2:
        print(f"  [ERROR] Fisher Exact Test requires a 2×2 table.")
        print(f"          Current table is {n_rows}×{n_cols}.")
        return None

    n_total = int(observed.sum())
    if n_total == 0:
        print("  [ERROR] Contingency table is empty (total N = 0).")
        return None

    a = int(observed[0, 0])
    b = int(observed[0, 1])
    c = int(observed[1, 0])
    d = int(observed[1, 1])

    try:
        or_stat, p_value = fisher_exact(observed, alternative="two-sided")
    except ValueError as exc:
        print(f"  [ERROR] Fisher Exact Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Fisher Exact Test.\n          Details: {exc}")
        return None

    if math.isnan(p_value):
        print("  [ERROR] Fisher Exact Test returned NaN p-value.")
        return None

    # Handle infinite or NaN Odds Ratio from scipy
    if math.isnan(or_stat):
        or_stat = float("nan")
    if math.isinf(or_stat):
        or_stat = float("inf")

    # Compute 95% CI via Woolf logit method (handles zeros via +0.5 correction)
    or_ci, ci_lower, ci_upper = odds_ratio_ci(a, b, c, d)

    # Use scipy OR for display; use Woolf OR only if scipy returned inf/nan
    display_or = or_stat
    if math.isnan(or_stat) or math.isinf(or_stat):
        display_or = or_ci

    magnitude, direction = interpret_odds_ratio(display_or)

    return {
        "or_stat"     : display_or,
        "or_raw"      : float(or_stat),
        "p_value"     : float(p_value),
        "n_total"     : n_total,
        "ci_lower"    : ci_lower,
        "ci_upper"    : ci_upper,
        "cells"       : {"a": a, "b": b, "c": c, "d": d},
        "zero_cells"  : int(np.sum(observed == 0)),
        "magnitude"   : magnitude,
        "direction"   : direction,
        "row_labels"  : list(table.index),
        "col_labels"  : list(table.columns),
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, var1: str, var2: str):
    print_section("FISHER EXACT TEST RESULTS")

    sig = results["p_value"] < 0.05
    rl = results["row_labels"]
    cl = results["col_labels"]

    print(f"\n  Test             : Fisher Exact Test (Two-Sided)")
    print(f"  Variable 1       : {var1}  (Categories: {rl[0]}, {rl[1]})")
    print(f"  Variable 2       : {var2}  (Categories: {cl[0]}, {cl[1]})")
    print(f"  Null Hypothesis  : '{var1}' and '{var2}' are independent.")
    print(f"  Alt. Hypothesis  : '{var1}' and '{var2}' are associated.")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Odds Ratio (OR)':<40} {fmt_val(results['or_stat']):>20}")
    print(f"  {'95% CI Lower':<40} {fmt_val(results['ci_lower']):>20}")
    print(f"  {'95% CI Upper':<40} {fmt_val(results['ci_upper']):>20}")
    print(f"  {'p-value (two-sided, exact)':<40} {results['p_value']:>20.4f}")
    print(f"  {'Total Sample Size (N)':<40} {results['n_total']:>20,}")
    print(f"  {'Zero cells in table':<40} {results['zero_cells']:>20}")
    print(f"  {SEPARATOR2[:68]}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result  : {sig_str}")
    print(f"  p-value : {results['p_value']:.4f}  (α = 0.05)")


def display_odds_ratio(results: dict, var1: str, var2: str):
    print_section("ODDS RATIO & CONFIDENCE INTERVAL")

    or_val = results["or_stat"]
    ci_lo = results["ci_lower"]
    ci_hi = results["ci_upper"]
    rl = results["row_labels"]
    cl = results["col_labels"]

    print(f"\n  Odds Ratio Definition:")
    print(f"    OR = (a × d) / (b × c)")
    print(f"    Where:")
    print(f"      a = {var1}={rl[0]} AND {var2}={cl[0]}  : {results['cells']['a']:,}")
    print(f"      b = {var1}={rl[0]} AND {var2}={cl[1]}  : {results['cells']['b']:,}")
    print(f"      c = {var1}={rl[1]} AND {var2}={cl[0]}  : {results['cells']['c']:,}")
    print(f"      d = {var1}={rl[1]} AND {var2}={cl[1]}  : {results['cells']['d']:,}")

    print(f"\n  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'Odds Ratio (OR)':<40} {fmt_val(or_val):>20}")
    print(f"  {'95% CI Lower Bound':<40} {fmt_val(ci_lo):>20}")
    print(f"  {'95% CI Upper Bound':<40} {fmt_val(ci_hi):>20}")
    print(f"  {'OR Magnitude':<40} {results['magnitude']:>20}")
    print(f"  {'OR Direction':<40}")
    print(f"      {results['direction']}")

    print(f"\n  Odds Ratio Interpretation Guidelines:")
    print(f"    OR = 1.0        → No association between variables")
    print(f"    OR 0.5 – 2.0   → Weak association")
    print(f"    OR 2.0 – 5.0   → Moderate association")
    print(f"    OR > 5.0       → Strong association")
    print(f"    OR > 1         → Positive association (increased odds in row 1)")
    print(f"    OR < 1         → Negative association (decreased odds in row 1)")

    print(f"\n  Confidence Interval Interpretation:")
    if not math.isnan(ci_lo) and not math.isnan(ci_hi):
        if ci_lo > 1.0:
            print(f"    95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}] lies entirely above 1.0.")
            print(f"    This supports a statistically significant positive association.")
        elif ci_hi < 1.0:
            print(f"    95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}] lies entirely below 1.0.")
            print(f"    This supports a statistically significant negative association.")
        else:
            print(f"    95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}] crosses 1.0.")
            print(f"    This indicates no statistically significant association.")
    else:
        print(f"    95% CI could not be computed (possibly due to zero cells).")

    if results["zero_cells"] > 0:
        print(f"\n  [NOTE] Continuity correction (+0.5) applied to all cells for CI")
        print(f"         calculation due to {results['zero_cells']} zero cell(s) in the table.")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict, var1: str, var2: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    p = results["p_value"]
    or_val = results["or_stat"]
    ci_lo = results["ci_lower"]
    ci_hi = results["ci_upper"]
    sig = p < 0.05
    rl = results["row_labels"]
    cl = results["col_labels"]

    print(f"\n  VARIABLE 1        : {var1}  ({rl[0]} vs {rl[1]})")
    print(f"  VARIABLE 2        : {var2}  ({cl[0]} vs {cl[1]})")
    print(f"  TOTAL SAMPLE SIZE : {results['n_total']:,}")

    print(f"\n  WHAT DOES THIS TEST MEASURE?")
    print(f"  The Fisher Exact Test determines whether there is a statistically")
    print(f"  significant association between two binary categorical variables.")
    print(f"  Unlike the Chi-Square Test, it computes an exact p-value based on")
    print(f"  the hypergeometric distribution, making it reliable for small samples")
    print(f"  and sparse tables where Chi-Square approximations break down.")
    print(f"  The Odds Ratio quantifies the strength and direction of association.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        print(f"  There is a statistically significant association between")
        print(f"  '{var1}' and '{var2}'.")
        print(f"  (OR = {fmt_val(or_val)}, 95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}], p = {p:.4f})")
        print(f"\n  The distribution of '{var2}' differs significantly across")
        print(f"  the two categories of '{var1}'.")
    else:
        print(f"  No statistically significant association was detected between")
        print(f"  '{var1}' and '{var2}'.")
        print(f"  (OR = {fmt_val(or_val)}, 95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}], p = {p:.4f})")
        print(f"\n  The distribution of '{var2}' does not differ significantly")
        print(f"  across the two categories of '{var1}'.")
        print(f"  The two variables appear to be statistically independent.")

    print(f"\n  ODDS RATIO INTERPRETATION:")
    print(f"  OR = {fmt_val(or_val)} → {results['magnitude']} effect")
    print(f"  {results['direction']}")
    if not math.isnan(or_val) and not math.isinf(or_val):
        if or_val > 1.0:
            print(f"  Individuals in the '{rl[0]}' category of '{var1}' have")
            print(f"  {or_val:.2f}× higher odds of being in the '{cl[0]}' category")
            print(f"  of '{var2}' compared to the '{rl[1]}' category.")
        elif or_val < 1.0 and or_val > 0:
            inv_or = 1.0 / or_val
            print(f"  Individuals in the '{rl[0]}' category of '{var1}' have")
            print(f"  {inv_or:.2f}× lower odds of being in the '{cl[0]}' category")
            print(f"  of '{var2}' compared to the '{rl[1]}' category.")
        else:
            print(f"  The Odds Ratio of 1.0 indicates equal odds across groups.")

    print(f"\n  CONFIDENCE INTERVAL:")
    if not math.isnan(ci_lo) and not math.isnan(ci_hi):
        print(f"  We are 95% confident the true population OR lies between")
        print(f"  {fmt_val(ci_lo)} and {fmt_val(ci_hi)}.")
        if ci_lo > 1.0:
            print(f"  Since the entire CI is above 1.0, the association is")
            print(f"  statistically significant in the positive direction.")
        elif ci_hi < 1.0:
            print(f"  Since the entire CI is below 1.0, the association is")
            print(f"  statistically significant in the negative direction.")
        else:
            print(f"  Since the CI crosses 1.0, the association is not")
            print(f"  statistically significant at the 0.05 level.")
    else:
        print(f"  Confidence interval could not be reliably computed.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance does not establish clinical importance.")
    print(f"  An OR close to 1.0 may be statistically significant in large samples")
    print(f"  but clinically meaningless. Conversely, a large OR in a small sample")
    print(f"  may not reach significance despite clinical importance.")
    print(f"  Always interpret OR in the context of the clinical question,")
    print(f"  baseline event rates, and patient-specific factors.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Fisher Exact Test assumes independence of all observations.")
    print(f"  • It is appropriate only for 2×2 contingency tables.")
    print(f"  • This test does not adjust for confounding variables.")
    print(f"  • This is an observational analysis; causation cannot be inferred.")
    print(f"  • The Odds Ratio approximates Relative Risk only when the outcome")
    print(f"    is rare (< 10% prevalence); otherwise interpret with caution.")
    if results["zero_cells"] > 0:
        print(f"  • Zero cells present: OR and CI computed with continuity correction.")
        print(f"    This correction may slightly affect point estimates.")
    if results["n_total"] < 20:
        print(f"  • Very small sample (N = {results['n_total']:,}). Results are highly")
        print(f"    sensitive to individual observations. Interpret with extreme caution.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(results: dict,
                 desc: dict,
                 table: pd.DataFrame,
                 var1: str,
                 var2: str,
                 dataset_path: str) -> str:
    p = results["p_value"]
    or_val = results["or_stat"]
    ci_lo = results["ci_lower"]
    ci_hi = results["ci_upper"]
    sig = p < 0.05
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rl = results["row_labels"]
    cl = results["col_labels"]
    n_total = results["n_total"]

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Fisher Exact Test")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    for var in [var1, var2]:
        d = desc[var]
        lines.append(f"  Variable             : {var}")
        lines.append(f"  Total Valid N        : {d['total']:,}")
        lines.append(f"  Categories (2)       : {d['categories']}")
        lines.append(f"  {'Category':<30} {'Count':>10} {'Percent':>10}")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*10}")
        for cat, cnt, pct in zip(d["categories"], d["counts"], d["proportions"]):
            lines.append(f"  {str(cat):<30} {cnt:>10,} {pct:>9.2f}%")
        lines.append("")

    # Contingency Table
    col_labels = [str(c) for c in table.columns]
    row_labels_t = [str(r) for r in table.index]
    col_w = max(14, max(len(c) for c in col_labels) + 2)
    row_label_w = max(22, max(len(r) for r in row_labels_t) + 4)

    lines.append("  CONTINGENCY TABLE (Observed Frequencies)")
    lines.append(SEPARATOR2)
    lines.append(f"  Row variable    : {var1}")
    lines.append(f"  Column variable : {var2}")
    lines.append(f"  Table Dimension : 2 × 2")
    lines.append(f"  Total N         : {n_total:,}")
    lines.append("")
    hdr = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr += f" {c[:col_w]:>{col_w}}"
    hdr += f" {'Row Total':>{col_w}}"
    lines.append(hdr)
    lines.append(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    for row_lbl in row_labels_t:
        row_vals = table.loc[row_lbl]
        row_total = int(row_vals.sum())
        row_str = f"  {str(row_lbl)[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            row_str += f" {int(val):>{col_w},}"
        row_str += f" {row_total:>{col_w},}"
        lines.append(row_str)
    col_totals = table.sum(axis=0)
    total_str = f"  {'Col Total'[:row_label_w]:<{row_label_w}}"
    for val in col_totals:
        total_str += f" {int(val):>{col_w},}"
    total_str += f" {n_total:>{col_w},}"
    lines.append(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    lines.append(total_str)
    lines.append("")
    lines.append("  Cell Reference Layout:")
    lines.append(f"    a = {var1}={rl[0]} AND {var2}={cl[0]} : {results['cells']['a']:,}")
    lines.append(f"    b = {var1}={rl[0]} AND {var2}={cl[1]} : {results['cells']['b']:,}")
    lines.append(f"    c = {var1}={rl[1]} AND {var2}={cl[0]} : {results['cells']['c']:,}")
    lines.append(f"    d = {var1}={rl[1]} AND {var2}={cl[1]} : {results['cells']['d']:,}")

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append("  [1] Categorical Variables             : CONFIRMED (both binary)")
    lines.append(f"  [2] Exactly 2 categories per variable : CONFIRMED (2×2 table)")
    lines.append("  [3] Independence of Observations      : ASSUMED")
    lines.append("  [4] No minimum expected frequency     : ADVANTAGE of Fisher Exact Test")
    zero_cells = results["zero_cells"]
    lines.append(f"  [5] Zero cells in table               : {zero_cells} cell(s)")
    if zero_cells > 0:
        lines.append("       Continuity correction (+0.5) applied for OR CI calculation.")

    # Test Results
    lines.append("")
    lines.append("  FISHER EXACT TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append("  Test               : Fisher Exact Test (Two-Sided)")
    lines.append(f"  Variable 1         : {var1}  ({rl[0]} vs {rl[1]})")
    lines.append(f"  Variable 2         : {var2}  ({cl[0]} vs {cl[1]})")
    lines.append(f"  Odds Ratio (OR)    : {fmt_val(or_val)}")
    lines.append(f"  95% CI             : [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}]")
    lines.append(f"  p-value (exact)    : {p:.4f}")
    lines.append(f"  Total N            : {n_total:,}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result             : {sig_str}")

    # Odds Ratio & CI
    lines.append("")
    lines.append("  ODDS RATIO & CONFIDENCE INTERVAL")
    lines.append(SEPARATOR2)
    lines.append(f"  Odds Ratio (OR)        : {fmt_val(or_val)}")
    lines.append(f"  95% CI Lower Bound     : {fmt_val(ci_lo)}")
    lines.append(f"  95% CI Upper Bound     : {fmt_val(ci_hi)}")
    lines.append(f"  OR Magnitude           : {results['magnitude']}")
    lines.append(f"  OR Direction           : {results['direction']}")
    lines.append(f"  CI Method              : Woolf logit method")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append("  There is a statistically significant association between the two categorical variables.")
        lines.append(f"  OR = {fmt_val(or_val)}, 95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}], p = {p:.4f}")
    else:
        lines.append("  No statistically significant association was detected.")
        lines.append(f"  OR = {fmt_val(or_val)}, 95% CI [{fmt_val(ci_lo)}, {fmt_val(ci_hi)}], p = {p:.4f}")
    lines.append("")
    lines.append("  The Fisher Exact Test determines whether two binary categorical variables")
    lines.append("  are associated. It uses an exact hypergeometric distribution method,")
    lines.append("  making it reliable for small and sparse datasets where Chi-Square fails.")
    lines.append(f"  OR = {fmt_val(or_val)} ({results['magnitude']} magnitude). {results['direction']}.")
    lines.append("  Statistical significance does not imply causation.")
    lines.append("  Confounding variables have not been adjusted for in this analysis.")

    lines.append("")
    lines.append(SEPARATOR)
    lines.append("  END OF REPORT")
    lines.append(SEPARATOR)
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str):
    print_section("STEP 9 | SAVE REPORT")
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

def run_fisher_exact_test():
    print_header()
    print("  This module performs the Fisher Exact Test.")
    print("  It tests whether two binary categorical variables are associated.")
    print("  Preferred over Chi-Square for small samples and sparse tables.")
    print("  Both variables must contain exactly 2 categories (binary).")

    # Step 1: Load CSV
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select Variable 1
    print_section("STEP 3 | SELECT VARIABLE 1")
    var1 = select_binary_column(df, "Variable 1")
    if var1 is None:
        sys.exit(1)

    # Step 4: Select Variable 2
    print_section("STEP 4 | SELECT VARIABLE 2")
    var2 = select_binary_column(df, "Variable 2", exclude_col=var1)
    if var2 is None:
        sys.exit(1)

    # Step 5: Clean data
    clean_df = clean_data(df, var1, var2)
    if clean_df is None:
        sys.exit(1)

    # Step 6: Descriptive statistics
    desc = compute_descriptive_stats(clean_df, var1, var2)
    display_descriptive_stats(desc, var1, var2)

    # Step 7: Build contingency table
    table = build_contingency_table(clean_df, var1, var2)

    if table.empty or table.values.sum() == 0:
        print("\n  [FATAL] Contingency table is empty. Cannot proceed.")
        sys.exit(1)

    # Final check: must be 2×2
    if table.shape != (2, 2):
        print(f"\n  [FATAL] Resulting table is {table.shape[0]}×{table.shape[1]}, not 2×2.")
        print("          Fisher Exact Test requires exactly a 2×2 contingency table.")
        print("          Please verify your variable selections and data.")
        sys.exit(1)

    display_contingency_table(table, var1, var2)

    # Step 8: Assumption checks
    display_assumption_checks(table, var1, var2)

    # Step 9: Run Fisher Exact Test
    print_section("STEP 6 | RUNNING FISHER EXACT TEST")
    results = run_fisher_exact(table)
    if results is None:
        print("\n  [FATAL] Fisher Exact Test could not be completed.")
        sys.exit(1)

    # Step 10: Display results
    display_test_results(results, var1, var2)

    # Step 11: Odds Ratio & CI
    display_odds_ratio(results, var1, var2)

    # Step 12: Medical interpretation
    display_interpretation(results, var1, var2)

    # Step 13: Build and save report
    report_text = build_report(results, desc, table, var1, var2, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Fisher Exact Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_fisher_exact_test()