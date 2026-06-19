
import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")



SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "chi_square_test_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Chi-Square Test of Independence
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
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, float):
        return f"{val:,.4f}"
    return str(val)


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """
    Compute Cramer's V as the effect size for Chi-Square Test.

        V = sqrt( χ² / (N × min(r−1, c−1)) )

    where r = number of rows, c = number of columns, N = total observations.
    Returns NaN when the denominator is zero.
    """
    min_dim = min(n_rows - 1, n_cols - 1)
    if min_dim <= 0 or n <= 0:
        return float("nan")
    return math.sqrt(chi2 / (n * min_dim))


def interpret_cramers_v(v: float) -> str:
    """
    Interpret Cramer's V using standard benchmarks:
      V < 0.10 → Negligible
      V < 0.30 → Small
      V < 0.50 → Medium
      V ≥ 0.50 → Large
    """
    if math.isnan(v):
        return "N/A"
    if v < 0.10:
        return "Negligible"
    elif v < 0.30:
        return "Small"
    elif v < 0.50:
        return "Medium"
    else:
        return "Large"


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

def select_categorical_column(df: pd.DataFrame,
                               label: str,
                               exclude_col: str | None = None) -> str | None:
    """
    Prompt user to select a categorical column.
    Validates that the column exists and has at least 2 unique categories.
    """
    print(f"\n  The {label} must be a categorical column.")
    print(f"  Example: sex, race, death, dnr, diabetes, treatment_group, etc.")

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
            print(f"  [ERROR] Variable 2 cannot be the same as Variable 1 ('{exclude_col}').")
            continue

        valid_values = df[col].dropna()
        unique_cats = valid_values.unique()
        n_unique = len(unique_cats)

        if n_unique == 0:
            print(f"  [ERROR] Column '{col}' contains no non-missing values.")
            continue
        if n_unique < 2:
            print(f"  [ERROR] Column '{col}' has only {n_unique} unique category: {list(unique_cats)}")
            print("          Chi-Square Test requires at least 2 categories per variable.")
            continue

        print(f"  [OK] {label} selected: '{col}'")
        print(f"       Categories ({n_unique}): {list(unique_cats[:10])}"
              + (" ..." if n_unique > 10 else ""))
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

    # Convert to string to ensure categorical treatment; strip whitespace
    working[var1] = working[var1].astype(str).str.strip()
    working[var2] = working[var2].astype(str).str.strip()

    # Remove rows that became empty string after stripping
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
        print("          Cannot proceed with analysis.")
        return None

    # Re-check categories after cleaning
    cats1 = working[var1].unique()
    cats2 = working[var2].unique()

    if len(cats1) < 2:
        print(f"\n  [ERROR] After cleaning, '{var1}' has only {len(cats1)} category: {list(cats1)}")
        print("          Chi-Square Test requires at least 2 categories.")
        return None
    if len(cats2) < 2:
        print(f"\n  [ERROR] After cleaning, '{var2}' has only {len(cats2)} category: {list(cats2)}")
        print("          Chi-Square Test requires at least 2 categories.")
        return None

    print(f"\n  [OK] Data is clean and ready for analysis.")
    return working


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptive_stats(df: pd.DataFrame,
                               var1: str,
                               var2: str) -> dict:
    """Compute category counts and proportions for each variable."""
    desc = {}
    for var in [var1, var2]:
        counts = df[var].value_counts().sort_index()
        total = counts.sum()
        desc[var] = {
            "categories"  : list(counts.index),
            "counts"      : list(counts.values),
            "proportions" : [round(c / total * 100, 2) for c in counts.values],
            "n_categories": len(counts),
            "total"       : int(total),
        }
    return desc


def display_descriptive_stats(desc: dict, var1: str, var2: str):
    print_section("DESCRIPTIVE STATISTICS")

    for var in [var1, var2]:
        d = desc[var]
        print(f"\n  Variable         : {var}")
        print(f"  Total Valid N    : {d['total']:,}")
        print(f"  Number of Categories: {d['n_categories']}")
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
    """Build observed frequency contingency table."""
    table = pd.crosstab(df[var1], df[var2])
    return table


def display_contingency_table(table: pd.DataFrame, var1: str, var2: str):
    print_section("CONTINGENCY TABLE (Observed Frequencies)")

    col_labels = [str(c) for c in table.columns]
    row_labels = [str(r) for r in table.index]
    n_total = int(table.values.sum())

    col_w = max(12, max(len(c) for c in col_labels) + 2)
    row_label_w = max(20, max(len(r) for r in row_labels) + 2)

    print(f"\n  Row variable    : {var1}")
    print(f"  Column variable : {var2}")
    print(f"  Total N         : {n_total:,}\n")

    # Header
    hdr = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr += f" {c[:col_w]:>{col_w}}"
    hdr += f" {'Row Total':>{col_w}}"
    print(hdr)
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))

    # Rows
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_total = int(row_vals.sum())
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            row_str += f" {int(val):>{col_w},}"
        row_str += f" {row_total:>{col_w},}"
        print(row_str)

    # Column totals
    col_totals = table.sum(axis=0)
    total_str = f"  {'Col Total'[:row_label_w]:<{row_label_w}}"
    for val in col_totals:
        total_str += f" {int(val):>{col_w},}"
    total_str += f" {n_total:>{col_w},}"
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    print(total_str)

    # Row percentages
    print(f"\n  ROW PERCENTAGES (% within {var1}):\n")
    hdr2 = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr2 += f" {c[:col_w]:>{col_w}}"
    print(hdr2)
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_total = row_vals.sum()
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            pct = (val / row_total * 100) if row_total > 0 else 0.0
            row_str += f" {pct:>{col_w}.1f}%"[:-1] + "%"
        print(row_str)

    # Column percentages
    print(f"\n  COLUMN PERCENTAGES (% within {var2}):\n")
    print(hdr2)
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        col_totals_arr = table.sum(axis=0)
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for col_lbl, val in zip(col_labels, row_vals):
            col_total = col_totals_arr[col_lbl] if col_lbl in col_totals_arr else 0
            pct = (val / col_total * 100) if col_total > 0 else 0.0
            row_str += f" {pct:>{col_w}.1f}%"[:-1] + "%"
        print(row_str)


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def check_assumptions(expected: np.ndarray) -> dict:
    """
    Check Chi-Square assumptions based on expected frequencies.
    Returns summary statistics about cells with expected count < 5.
    """
    total_cells = expected.size
    cells_below_5 = int(np.sum(expected < 5))
    pct_below_5 = (cells_below_5 / total_cells * 100) if total_cells > 0 else 0.0
    min_expected = float(np.min(expected))
    assumption_met = pct_below_5 <= 20.0

    return {
        "total_cells"   : total_cells,
        "cells_below_5" : cells_below_5,
        "pct_below_5"   : pct_below_5,
        "min_expected"  : min_expected,
        "assumption_met": assumption_met,
    }


def display_expected_table(expected: np.ndarray,
                            table: pd.DataFrame,
                            var1: str,
                            var2: str):
    """Display the expected frequencies table."""
    col_labels = [str(c) for c in table.columns]
    row_labels = [str(r) for r in table.index]

    col_w = max(12, max(len(c) for c in col_labels) + 2)
    row_label_w = max(20, max(len(r) for r in row_labels) + 2)

    print(f"\n  EXPECTED FREQUENCIES TABLE:\n")
    hdr = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr += f" {c[:col_w]:>{col_w}}"
    print(hdr)
    print(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    for i, row_lbl in enumerate(row_labels):
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for j in range(len(col_labels)):
            val = expected[i, j]
            marker = " *" if val < 5 else "  "
            row_str += f" {val:>{col_w-2}.2f}{marker}"
        print(row_str)
    print(f"\n  * = Expected frequency < 5")


def display_assumption_checks(assump: dict, var1: str, var2: str):
    print_section("ASSUMPTION CHECKS")

    print("\n  [1] Independence of Observations")
    print("      ASSUMED: Each observation is independent of all others.")
    print("      Ensure data collection followed independent sampling design.")
    print("      Chi-Square Test is NOT appropriate for paired/repeated measures.")

    print("\n  [2] Categorical Scale of Measurement")
    print(f"      ASSUMED: Both variables are categorical (nominal or ordinal).")
    print(f"      Variable 1 ({var1}) and Variable 2 ({var2}) are treated as categorical.")

    print("\n  [3] Adequate Expected Frequencies (Cochran's Rule)")
    print(f"      Rule: No more than 20% of cells should have expected count < 5.")
    print(f"      All cells should ideally have expected count ≥ 1.\n")
    print(f"      Total cells              : {assump['total_cells']}")
    print(f"      Cells with expected < 5  : {assump['cells_below_5']}")
    print(f"      Percentage below 5       : {assump['pct_below_5']:.1f}%")
    print(f"      Minimum expected count   : {assump['min_expected']:.4f}")

    if assump["assumption_met"]:
        print(f"\n      [OK] Expected frequency assumption is satisfied.")
        print(f"           Chi-Square Test results are reliable.")
    else:
        print(f"\n      [WARNING] Chi-Square assumptions may be violated.")
        print(f"               More than 20% of cells have expected count < 5.")
        print(f"               Consider Fisher's Exact Test for more reliable results.")
        print(f"               Interpret Chi-Square results with caution.")

    print("\n  [4] Sample Size")
    print("      Chi-Square Test is appropriate for large samples.")
    print("      For small samples (N < 20), consider Fisher's Exact Test.")

    print("\n  [5] Why Chi-Square Test of Independence?")
    print("      The Chi-Square Test was selected because:")
    print("      - Both variables are categorical.")
    print("      - It tests whether the distribution of one variable")
    print("        differs across categories of the other variable.")
    print("      - It is the standard test for association between")
    print("        two categorical variables in medical research.")


# ======================================================================
# CHI-SQUARE TEST
# ======================================================================

def run_chi_square(table: pd.DataFrame) -> dict | None:
    """
    Run Chi-Square Test of Independence using scipy.stats.chi2_contingency().
    Returns a dictionary of all computed statistics, or None on failure.
    """
    observed = table.values.astype(float)
    n_rows, n_cols = observed.shape
    n_total = int(observed.sum())

    if n_total == 0:
        print("  [ERROR] Contingency table is empty (total N = 0).")
        return None

    if n_rows < 2 or n_cols < 2:
        print(f"  [ERROR] Contingency table has insufficient dimensions: {n_rows}×{n_cols}.")
        print("          Both variables must have at least 2 categories.")
        return None

    try:
        chi2, p_value, dof, expected = chi2_contingency(observed, correction=False)
    except ValueError as exc:
        print(f"  [ERROR] Chi-Square Test failed.\n          Details: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during Chi-Square Test.\n          Details: {exc}")
        return None

    if math.isnan(chi2) or math.isnan(p_value):
        print("  [ERROR] Chi-Square Test returned NaN results.")
        print("          This may indicate an empty or degenerate contingency table.")
        return None

    v = cramers_v(chi2, n_total, n_rows, n_cols)
    effect_label = interpret_cramers_v(v)
    assump = check_assumptions(expected)

    return {
        "chi2"         : float(chi2),
        "p_value"      : float(p_value),
        "dof"          : int(dof),
        "n_total"      : n_total,
        "n_rows"       : n_rows,
        "n_cols"       : n_cols,
        "expected"     : expected,
        "observed"     : observed,
        "v"            : float(v),
        "effect_label" : effect_label,
        "assump"       : assump,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_test_results(results: dict, var1: str, var2: str):
    print_section("CHI-SQUARE TEST RESULTS")

    sig = results["p_value"] < 0.05

    print(f"\n  Test             : Chi-Square Test of Independence")
    print(f"  Variable 1       : {var1}  ({results['n_rows']} categories)")
    print(f"  Variable 2       : {var2}  ({results['n_cols']} categories)")
    print(f"  Null Hypothesis  : '{var1}' and '{var2}' are independent.")
    print(f"  Alt. Hypothesis  : '{var1}' and '{var2}' are associated.")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Statistic':<40} {'Value':>20}")
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {'Chi-Square Statistic (χ²)':<40} {results['chi2']:>20.4f}")
    print(f"  {'Degrees of Freedom':<40} {results['dof']:>20}")
    print(f"  {'p-value':<40} {results['p_value']:>20.4f}")
    print(f"  {'Total Sample Size (N)':<40} {results['n_total']:>20,}")
    print(f"  {'Table Dimensions':<40} {str(results['n_rows']) + ' × ' + str(results['n_cols']):>20}")
    print(f"  {SEPARATOR2[:68]}")

    sig_str = "YES — Statistically Significant" if sig else "NO  — Not Statistically Significant"
    print(f"\n  Result  : {sig_str}")
    print(f"  p-value : {results['p_value']:.4f}  (α = 0.05)")

    if not results["assump"]["assumption_met"]:
        print(f"\n  [WARNING] Expected frequency assumption violated.")
        print(f"            {results['assump']['pct_below_5']:.1f}% of cells have expected count < 5.")
        print(f"            Consider Fisher's Exact Test for more reliable inference.")


def display_effect_size(results: dict):
    print_section("EFFECT SIZE")

    v = results["v"]
    print(f"\n  Effect Size Measure : Cramer's V")
    print(f"  Formula             : V = √( χ² / (N × min(r−1, c−1)) )\n")

    print(f"  {'Measure':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'Chi-Square (χ²)':<40} {results['chi2']:>20.4f}")
    print(f"  {'Total N':<40} {results['n_total']:>20,}")
    print(f"  {'min(r−1, c−1)':<40} {min(results['n_rows']-1, results['n_cols']-1):>20}")
    print(f"  {'Cramer\'s V':<40} {v:>20.4f}")
    print(f"  {'Effect Size Interpretation':<40} {results['effect_label']:>20}")

    print(f"\n  Effect Size Benchmarks (Cohen 1988):")
    print(f"    V < 0.10 → Negligible")
    print(f"    V < 0.30 → Small")
    print(f"    V < 0.50 → Medium")
    print(f"    V ≥ 0.50 → Large")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(results: dict,
                            desc: dict,
                            var1: str,
                            var2: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    p = results["p_value"]
    v = results["v"]
    chi2 = results["chi2"]
    dof = results["dof"]
    sig = p < 0.05

    print(f"\n  VARIABLE 1        : {var1}  ({results['n_rows']} categories)")
    print(f"  VARIABLE 2        : {var2}  ({results['n_cols']} categories)")
    print(f"  TOTAL SAMPLE SIZE : {results['n_total']:,}")

    print(f"\n  WHAT DOES THIS TEST MEASURE?")
    print(f"  The Chi-Square Test of Independence determines whether there is a")
    print(f"  statistically significant association between two categorical variables.")
    print(f"  It compares the observed frequencies in each cell of the contingency")
    print(f"  table to the frequencies expected if the variables were independent.")
    print(f"  A significant result means the distribution of one variable differs")
    print(f"  systematically across the categories of the other variable.")

    print(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        print(f"  There is a statistically significant association between")
        print(f"  '{var1}' and '{var2}'.")
        print(f"  (χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Cramer's V = {v:.4f})")
        print(f"\n  The observed distribution of '{var2}' categories differs")
        print(f"  significantly across the groups of '{var1}', indicating")
        print(f"  that the two variables are not independent.")
    else:
        print(f"  No statistically significant association was detected between")
        print(f"  '{var1}' and '{var2}'.")
        print(f"  (χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Cramer's V = {v:.4f})")
        print(f"\n  The observed distribution of '{var2}' categories does not differ")
        print(f"  significantly across the groups of '{var1}'.")
        print(f"  The two variables appear to be independent of each other.")

    print(f"\n  EFFECT SIZE COMMENTARY:")
    effect_desc = results["effect_label"]
    print(f"  Cramer's V (V = {v:.4f}) indicates a {effect_desc} association.")
    if v < 0.10:
        print(f"  The practical association between the variables is negligible.")
    elif v < 0.30:
        print(f"  There is a small but potentially meaningful association between variables.")
    elif v < 0.50:
        print(f"  There is a moderate association between the two variables.")
    else:
        print(f"  There is a large and practically important association between variables.")

    if not results["assump"]["assumption_met"]:
        print(f"\n  ASSUMPTION VIOLATION NOTE:")
        print(f"  {results['assump']['pct_below_5']:.1f}% of cells have expected count < 5.")
        print(f"  The Chi-Square approximation may be unreliable for this dataset.")
        print(f"  Fisher's Exact Test is recommended for more valid inference.")

    print(f"\n  CLINICAL RELEVANCE:")
    print(f"  Statistical significance alone does not establish clinical significance.")
    print(f"  An association between categorical variables does not imply causation.")
    print(f"  Clinicians should consider the clinical context, the magnitude of")
    print(f"  association (Cramer's V), sample representativeness, and potential")
    print(f"  confounders when interpreting these results.")

    print(f"\n  LIMITATIONS:")
    print(f"  • Chi-Square Test assumes independence of all observations.")
    print(f"  • Results may be unreliable when expected frequencies are small.")
    print(f"  • This test does not indicate the direction of the association.")
    print(f"  • Confounding variables have not been adjusted for in this analysis.")
    print(f"  • This is an observational analysis and does not establish causality.")
    if results["n_total"] < 20:
        print(f"  • Very small sample size (N = {results['n_total']:,}).")
        print(f"    Interpret results with extreme caution.")


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
    chi2 = results["chi2"]
    dof = results["dof"]
    v = results["v"]
    sig = p < 0.05
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assump = results["assump"]

    lines = []
    lines.append(SEPARATOR)
    lines.append("  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit")
    lines.append("  Module  : Chi-Square Test of Independence")
    lines.append(f"  Dataset : {dataset_path}")
    lines.append(f"  Date    : {ts}")
    lines.append(SEPARATOR)

    # Descriptive Statistics
    lines.append("")
    lines.append("  DESCRIPTIVE STATISTICS")
    lines.append(SEPARATOR2)
    for var in [var1, var2]:
        d = desc[var]
        lines.append(f"  Variable         : {var}")
        lines.append(f"  Total Valid N    : {d['total']:,}")
        lines.append(f"  Categories ({d['n_categories']}): {d['categories']}")
        lines.append(f"  {'Category':<30} {'Count':>10} {'Percent':>10}")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*10}")
        for cat, cnt, pct in zip(d["categories"], d["counts"], d["proportions"]):
            lines.append(f"  {str(cat):<30} {cnt:>10,} {pct:>9.2f}%")
        lines.append("")

    # Contingency Table
    lines.append("  CONTINGENCY TABLE (Observed Frequencies)")
    lines.append(SEPARATOR2)
    col_labels = [str(c) for c in table.columns]
    row_labels = [str(r) for r in table.index]
    col_w = max(12, max(len(c) for c in col_labels) + 2)
    row_label_w = max(20, max(len(r) for r in row_labels) + 2)
    n_total_rep = int(table.values.sum())

    hdr = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr += f" {c[:col_w]:>{col_w}}"
    hdr += f" {'Row Total':>{col_w}}"
    lines.append(hdr)
    lines.append(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    for row_lbl in row_labels:
        row_vals = table.loc[row_lbl]
        row_total = int(row_vals.sum())
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for val in row_vals:
            row_str += f" {int(val):>{col_w},}"
        row_str += f" {row_total:>{col_w},}"
        lines.append(row_str)
    col_totals = table.sum(axis=0)
    total_str = f"  {'Col Total'[:row_label_w]:<{row_label_w}}"
    for val in col_totals:
        total_str += f" {int(val):>{col_w},}"
    total_str += f" {n_total_rep:>{col_w},}"
    lines.append(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * (len(col_labels) + 1))
    lines.append(total_str)

    # Expected Frequencies
    lines.append("")
    lines.append("  EXPECTED FREQUENCIES TABLE")
    lines.append(SEPARATOR2)
    hdr2 = f"  {var1[:row_label_w]:<{row_label_w}}"
    for c in col_labels:
        hdr2 += f" {c[:col_w]:>{col_w}}"
    lines.append(hdr2)
    lines.append(f"  {'-'*row_label_w}" + f" {'-'*col_w}" * len(col_labels))
    for i, row_lbl in enumerate(row_labels):
        row_str = f"  {row_lbl[:row_label_w]:<{row_label_w}}"
        for j in range(len(col_labels)):
            val = results["expected"][i, j]
            marker = "*" if val < 5 else " "
            row_str += f" {val:>{col_w-1}.2f}{marker}"
        lines.append(row_str)
    lines.append(f"  * = Expected frequency < 5")

    # Assumption Checks
    lines.append("")
    lines.append("  ASSUMPTION CHECKS")
    lines.append(SEPARATOR2)
    lines.append("  [1] Independence of Observations: ASSUMED")
    lines.append("  [2] Categorical variables: CONFIRMED")
    lines.append(f"  [3] Expected frequencies (Cochran's Rule):")
    lines.append(f"       Total cells              : {assump['total_cells']}")
    lines.append(f"       Cells with expected < 5  : {assump['cells_below_5']}")
    lines.append(f"       Percentage below 5       : {assump['pct_below_5']:.1f}%")
    lines.append(f"       Minimum expected count   : {assump['min_expected']:.4f}")
    if assump["assumption_met"]:
        lines.append("       Result: ASSUMPTION SATISFIED")
    else:
        lines.append("       Result: ASSUMPTION VIOLATED — Consider Fisher's Exact Test")

    # Test Results
    lines.append("")
    lines.append("  CHI-SQUARE TEST RESULTS")
    lines.append(SEPARATOR2)
    lines.append(f"  Test               : Chi-Square Test of Independence")
    lines.append(f"  Variable 1         : {var1}  ({results['n_rows']} categories)")
    lines.append(f"  Variable 2         : {var2}  ({results['n_cols']} categories)")
    lines.append(f"  Chi-Square (χ²)    : {chi2:.4f}")
    lines.append(f"  Degrees of Freedom : {dof}")
    lines.append(f"  p-value            : {p:.4f}")
    lines.append(f"  Total N            : {results['n_total']:,}")
    sig_str = "SIGNIFICANT (p < 0.05)" if sig else "NOT SIGNIFICANT (p ≥ 0.05)"
    lines.append(f"  Result             : {sig_str}")

    # Effect Size
    lines.append("")
    lines.append("  EFFECT SIZE")
    lines.append(SEPARATOR2)
    lines.append(f"  Cramer's V                     : {v:.4f}")
    lines.append(f"  Effect Size Interpretation     : {results['effect_label']}")
    lines.append(f"  Formula                        : V = √( χ² / (N × min(r−1, c−1)) )")

    # Interpretation
    lines.append("")
    lines.append("  MEDICAL & STATISTICAL INTERPRETATION")
    lines.append(SEPARATOR2)
    if sig:
        lines.append(f"  There is a statistically significant association between the variables.")
        lines.append(f"  χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Cramer's V = {v:.4f}")
    else:
        lines.append(f"  No statistically significant association was detected.")
        lines.append(f"  χ²({dof}) = {chi2:.4f}, p = {p:.4f}, Cramer's V = {v:.4f}")
    lines.append("")
    lines.append("  The Chi-Square Test of Independence tests whether two categorical")
    lines.append("  variables are associated or independent. A significant result indicates")
    lines.append("  that the distribution of one variable differs across categories of the other.")
    lines.append(f"  Effect size (Cramer's V = {v:.4f}) is classified as: {results['effect_label']}.")
    lines.append("  Statistical significance does not imply causation.")
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

def run_chi_square_test():
    print_header()
    print("  This module performs the Chi-Square Test of Independence.")
    print("  It tests whether two categorical variables are associated.")
    print("  Appropriate for nominal and ordinal categorical data.")

    # Step 1: Load CSV
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3 & 4: Select variables
    print_section("STEP 3 | SELECT VARIABLE 1")
    var1 = select_categorical_column(df, "Variable 1")
    if var1 is None:
        sys.exit(1)

    print_section("STEP 4 | SELECT VARIABLE 2")
    var2 = select_categorical_column(df, "Variable 2", exclude_col=var1)
    if var2 is None:
        sys.exit(1)

    # Step 5: Clean data
    clean_df = clean_data(df, var1, var2)
    if clean_df is None:
        sys.exit(1)

    # Step 6: Descriptive statistics
    desc = compute_descriptive_stats(clean_df, var1, var2)
    display_descriptive_stats(desc, var1, var2)

    # Step 7: Contingency table
    table = build_contingency_table(clean_df, var1, var2)

    if table.empty or table.values.sum() == 0:
        print("\n  [FATAL] Contingency table is empty. Cannot proceed.")
        sys.exit(1)

    display_contingency_table(table, var1, var2)

    # Step 8: Assumption checks (need expected frequencies first)
    print_section("STEP 6 | RUNNING CHI-SQUARE TEST")
    results = run_chi_square(table)
    if results is None:
        print("\n  [FATAL] Chi-Square Test could not be completed.")
        sys.exit(1)

    # Display expected table within assumption checks
    display_assumption_checks(results["assump"], var1, var2)
    display_expected_table(results["expected"], table, var1, var2)

    # Step 9: Test results
    display_test_results(results, var1, var2)

    # Step 10: Effect size
    display_effect_size(results)

    # Step 11: Medical interpretation
    display_interpretation(results, desc, var1, var2)

    # Step 12: Build and save report
    report_text = build_report(results, desc, table, var1, var2, path)
    save_report(report_text)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Chi-Square Test — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_chi_square_test()