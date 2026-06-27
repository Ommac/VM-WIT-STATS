"""
======================================================================
VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
======================================================================
Module      : Kendall's Tau Correlation
File        : correlation/kendall_tau.py
Description : Measures the strength and direction of monotonic
              association between two variables using Kendall's Tau-b.

              Uses scipy.stats.kendalltau(), which computes Kendall's
              Tau-b and automatically corrects for tied ranks.

WHEN TO USE:
    - Ordinal variables (e.g. pain score, disease stage, Likert scale)
    - Ranked data
    - Small samples
    - Data with many tied ranks
    - Non-normally distributed continuous variables

WHEN NOT TO USE:
    - Both variables are nominal/categorical (use Chi-Square or Cramér's V)
    - You need to measure a linear relationship (use Pearson)
    - You are primarily interested in direction and linearity

ASSUMPTIONS:
    1. Two variables measured on at least an ordinal scale.
    2. Observations are independent.
    3. Monotonic (not necessarily linear) relationship assumed under H1.

REQUIRED LIBRARIES:
    pip install pandas numpy scipy
======================================================================
"""

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
REPORT_FILE = os.path.join(REPORT_DIR, "kendall_tau_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Kendall's Tau Correlation
{SEPARATOR}
"""

# Suggested interpretation thresholds for |τ|.
# These are guidelines for medical research context.
STRENGTH_THRESHOLDS = [
    (0.10, "Negligible"),
    (0.30, "Weak"),
    (0.50, "Moderate"),
    (0.70, "Strong"),
    (1.01, "Very Strong"),
]


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_header() -> None:
    print(HEADER)


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR2}")
    print(f"  {title}")
    print(SEPARATOR2)


def fmt(val, decimals: int = 4) -> str:
    """Format a scalar for display, handling None, NaN, and Inf."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        return f"{val:.{decimals}f}"
    return str(val)


def interpret_strength(tau: float) -> str:
    """
    Return the strength label for a given |τ| value.

    Suggested interpretation of Kendall's Tau magnitude:
        |τ| < 0.10  → Negligible
        0.10–0.29   → Weak
        0.30–0.49   → Moderate
        0.50–0.69   → Strong
        ≥ 0.70      → Very Strong
    """
    abs_tau = abs(tau)
    for threshold, label in STRENGTH_THRESHOLDS:
        if abs_tau < threshold:
            return label
    return "Very Strong"


def interpret_direction(tau: float) -> str:
    """Return 'Positive', 'Negative', or 'None' for a tau value."""
    if abs(tau) < 1e-10:
        return "None (no association)"
    return "Positive" if tau > 0 else "Negative"


def interpret_p(p: float) -> str:
    """Return a graded significance statement for a p-value."""
    if p < 0.001:
        return "Highly statistically significant (p < 0.001)"
    elif p < 0.01:
        return "Statistically significant (p < 0.01)"
    elif p < 0.05:
        return "Statistically significant (p < 0.05)"
    else:
        return "Not statistically significant (p ≥ 0.05)"



# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    """Prompt user for CSV file path with validation loop."""
    print_section("STEP 1 | LOAD DATASET")
    print("\n  NOTE: Kendall's Tau requires two numeric or ordinal variables.")
    print("        Each row must represent one independent observation.")

    while True:
        path = input("\n  Enter path to CSV file: ").strip()
        if not path:
            print("  [ERROR] No path entered. Please try again.")
            continue
        if not os.path.isfile(path):
            print(f"  [ERROR] File not found: '{path}'.")
            continue
        return path


def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV safely with comprehensive exception handling."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print("  [ERROR] The CSV file is empty.")
        return None
    except pd.errors.ParserError as exc:
        print(f"  [ERROR] Failed to parse CSV: {exc}")
        return None
    except PermissionError:
        print("  [ERROR] Permission denied when reading the file.")
        return None
    except Exception as exc:
        print(f"  [ERROR] Unexpected error: {exc}")
        return None

    if df.empty:
        print("  [ERROR] CSV loaded but contains no rows.")
        return None

    print(f"\n  [OK] Dataset loaded.  Rows: {df.shape[0]:,}  Columns: {df.shape[1]:,}")
    return df


def display_columns(df: pd.DataFrame) -> None:
    """Print a numbered table of column names and dtypes."""
    print_section("STEP 2 | AVAILABLE COLUMNS")
    print(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15}")
    print(f"  {'-'*5} {'-'*40} {'-'*15}")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        print(f"  {idx:<5} {col:<40} {str(dtype):<15}")


# ======================================================================
# COLUMN SELECTION & VALIDATION
# ======================================================================

def select_column(df: pd.DataFrame,
                   label: str,
                   exclude: list[str] | None = None) -> str | None:
    """
    Prompt user to select one numeric column.

    Parameters
    ----------
    df      : DataFrame
    label   : Display label shown to user (e.g. 'first variable')
    exclude : Column names already selected (to prevent duplicate selection)

    Returns
    -------
    str : Column name, or None if a valid selection cannot be made
          after repeated attempts (loop is handled by caller).
    """
    exclude = exclude or []
    print(f"\n  Select the {label}.")
    print(f"  Must be numeric or ordinal (integer or float).")

    while True:
        col = input(f"\n  Enter column name for {label}: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue
        if col in exclude:
            print(f"  [ERROR] '{col}' is already selected. Choose a different column.")
            continue

        coerced   = pd.to_numeric(df[col], errors="coerce")
        valid_n   = int(coerced.notna().sum())
        if valid_n == 0:
            print(f"  [ERROR] Column '{col}' contains no numeric values.")
            continue

        non_num = len(df) - valid_n
        if non_num > 0:
            print(f"  [WARNING] {non_num:,} non-numeric value(s) in '{col}' "
                  f"will be treated as missing.")

        print(f"  [OK] '{col}' selected  (valid numeric values: {valid_n:,})")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_pair(df: pd.DataFrame,
               col_x: str,
               col_y: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extract, coerce, and clean the two selected columns.

    Steps:
        1. Coerce both columns to float64.
        2. Drop rows where either value is NaN or ±Inf.
        3. Enforce minimum sample size (n ≥ 3).
        4. Check for constant variables.
        5. Check that columns are not identical.

    Returns
    -------
    (x, y) : tuple of clean float64 arrays, or (None, None) on failure.
    """
    print_section("STEP 4 | DATA CLEANING & VALIDATION")

    x_raw = pd.to_numeric(df[col_x], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[col_y], errors="coerce").to_numpy(dtype=float)

    original_n = len(x_raw)
    print(f"\n  Original rows: {original_n:,}")

    # Drop NaN and ±Inf in either variable
    valid = np.isfinite(x_raw) & np.isfinite(y_raw)
    dropped = int((~valid).sum())
    x = x_raw[valid]
    y = y_raw[valid]
    n = len(x)

    if dropped > 0:
        print(f"  Dropped (missing/infinite): {dropped:,} row(s)")
    print(f"  Pairs used for analysis    : {n:,}")

    # Minimum sample size
    if n < 3:
        print(f"\n  [ERROR] Only {n} valid pair(s) remain. "
              f"Kendall's Tau requires at least 3 observations.")
        return None, None

    # Constant variable check
    if float(x.max()) - float(x.min()) < 1e-10:
        print(f"\n  [ERROR] '{col_x}' is constant (all values identical). "
              f"Tau is undefined for a constant variable.")
        return None, None
    if float(y.max()) - float(y.min()) < 1e-10:
        print(f"\n  [ERROR] '{col_y}' is constant (all values identical). "
              f"Tau is undefined for a constant variable.")
        return None, None

    # Identical columns
    if np.array_equal(x, y):
        print(f"\n  [WARNING] The two selected columns contain identical values. "
              f"τ will be 1.0 by definition. Verify your column selection.")

    print(f"\n  [OK] Data validated and ready for analysis.")
    return x, y


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptives(arr: np.ndarray, col: str) -> dict:
    """Compute basic descriptive statistics for one clean array."""
    n = len(arr)
    return {
        "col"    : col,
        "n"      : n,
        "mean"   : float(np.mean(arr)),
        "median" : float(np.median(arr)),
        "std"    : float(np.std(arr, ddof=1)) if n > 1 else float("nan"),
        "min"    : float(np.min(arr)),
        "max"    : float(np.max(arr)),
    }


def display_descriptives(dx: dict, dy: dict) -> None:
    """Print a side-by-side descriptive statistics table."""
    print_section("DESCRIPTIVE STATISTICS")

    label_w = 22
    val_w   = 20

    col_x = dx["col"]; col_y = dy["col"]
    header_x = str(col_x)[:val_w]
    header_y = str(col_y)[:val_w]

    print(f"\n  {'Statistic':<{label_w}} {header_x:>{val_w}} {header_y:>{val_w}}")
    print(f"  {'-'*label_w} {'-'*val_w} {'-'*val_w}")

    rows = [
        ("Sample Size (n)", "n"),
        ("Mean",            "mean"),
        ("Median",          "median"),
        ("Std Deviation",   "std"),
        ("Minimum",         "min"),
        ("Maximum",         "max"),
    ]

    for label, key in rows:
        vx = fmt(dx[key])
        vy = fmt(dy[key])
        print(f"  {label:<{label_w}} {vx:>{val_w}} {vy:>{val_w}}")


# ======================================================================
# KENDALL'S TAU COMPUTATION
# ======================================================================

def run_kendall_tau(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    Compute Kendall's Tau-b correlation.

    Uses scipy.stats.kendalltau(), which computes Kendall's Tau-b
    and automatically corrects for tied ranks.

    Parameters
    ----------
    x, y : clean float64 arrays of equal length

    Returns
    -------
    dict with keys: tau, p_value, n, direction, strength, significant
    Returns None if scipy returns NaN (e.g. both arrays constant).
    """
    n = len(x)

    try:
        result = stats.kendalltau(x, y)
    except Exception as exc:
        print(f"  [ERROR] scipy.stats.kendalltau failed: {exc}")
        return None

    tau     = float(result.statistic)
    p_value = float(result.pvalue)

    if math.isnan(tau) or math.isnan(p_value):
        print("  [ERROR] Kendall's Tau returned NaN. "
              "This may occur when one variable has zero variance.")
        return None

    # Clamp tau to [-1, 1] — guard against floating-point overshoot
    tau = max(-1.0, min(1.0, tau))

    direction  = interpret_direction(tau)
    strength   = interpret_strength(tau)
    # bool() ensures Python bool, never numpy.bool_
    significant = bool(p_value < 0.05)

    return {
        "tau"        : tau,
        "p_value"    : p_value,
        "n"          : n,
        "direction"  : direction,
        "strength"   : strength,
        "significant": significant,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_results(res: dict, col_x: str, col_y: str) -> None:
    """Print Kendall's Tau results in a formatted table."""
    print_section("KENDALL'S TAU RESULTS")

    tau   = res["tau"]
    p     = res["p_value"]
    n     = res["n"]
    sig   = res["significant"]

    print(f"\n  Variable 1       : {col_x}")
    print(f"  Variable 2       : {col_y}")
    print(f"  Test             : Kendall's Tau-b (two-tailed)")
    print(f"  Null Hypothesis  : No monotonic association between variables")
    print(f"  Alt. Hypothesis  : A monotonic association exists")
    print(f"  Significance (α) : 0.05\n")

    print(f"  {SEPARATOR2[:68]}")
    tau_label = "Kendall's Tau (τ)"
    print(f"  {tau_label:<40} {tau:>20.4f}")
    print(f"  {'p-value (two-tailed)':<40} {p:>20.4f}")
    print(f"  {'Sample Size (n)':<40} {n:>20,}")
    print(f"  {'Direction':<40} {res['direction']:>20}")
    print(f"  {'Strength':<40} {res['strength']:>20}")
    print(f"  {SEPARATOR2[:68]}")

    decision = "Reject H₀" if sig else "Fail to Reject H₀"
    print(f"\n  Decision : {decision}")
    print(f"  Result   : {interpret_p(p)}")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(res: dict, col_x: str, col_y: str) -> None:
    """Display statistical and medical interpretation of the result."""
    print_section("INTERPRETATION")

    tau = res["tau"]
    p   = res["p_value"]
    sig = res["significant"]
    n   = res["n"]

    direction = res["direction"]
    strength  = res["strength"]

    # Statistical interpretation
    print(f"\n  STATISTICAL INTERPRETATION:")
    if sig:
        print(f"  A statistically significant monotonic association was found")
        print(f"  between '{col_x}' and '{col_y}'.")
        print(f"  (τ = {tau:.4f}, p = {p:.4f})")
        if tau > 0:
            print(f"  As '{col_x}' increases, '{col_y}' tends to increase.")
        else:
            print(f"  As '{col_x}' increases, '{col_y}' tends to decrease.")
    else:
        print(f"  No statistically significant monotonic association was found")
        print(f"  between '{col_x}' and '{col_y}'.")
        print(f"  (τ = {tau:.4f}, p = {p:.4f})")

    print(f"\n  Association direction : {direction}")
    print(f"  Association strength  : {strength}")

    # Large sample advisory
    if n > 200 and sig:
        print(f"\n  [LARGE SAMPLE NOTE] n = {n:,}.")
        print(f"  With large samples, even small τ values can reach statistical")
        print(f"  significance. Evaluate the practical importance of τ = {tau:.4f}")
        print(f"  in the context of your research question.")

    # Medical interpretation
    print(f"\n  MEDICAL INTERPRETATION:")
    strength_lower = strength.lower()
    if sig:
        print(f"  The {strength_lower} {'positive' if tau > 0 else 'negative'} "
              f"association (τ = {tau:.4f}) between '{col_x}' and '{col_y}'")
        if tau > 0:
            print(f"  indicates that higher values of '{col_x}' are consistently")
            print(f"  associated with higher values of '{col_y}'.")
        else:
            print(f"  indicates that higher values of '{col_x}' are consistently")
            print(f"  associated with lower values of '{col_y}'.")
        print(f"\n  Statistical significance (p = {p:.4f}) confirms this association")
        print(f"  is unlikely to be due to chance alone.")
        print(f"  However, statistical significance does not imply clinical importance.")
        print(f"  Evaluate the magnitude of τ = {tau:.4f} in the context of your")
        print(f"  specific clinical or research question.")
    else:
        print(f"  No meaningful monotonic association was detected between")
        print(f"  '{col_x}' and '{col_y}' (τ = {tau:.4f}, p = {p:.4f}).")
        print(f"  The data do not support a consistent directional relationship")
        print(f"  between these two variables in this sample.")


def display_limitations() -> None:
    """Display limitations relevant to Kendall's Tau."""
    print_section("LIMITATIONS")

    print(f"\n  1. Correlation does NOT imply causation.")
    print(f"     An association between variables does not establish that")
    print(f"     one causes the other.")

    print(f"\n  2. Monotonic association only.")
    print(f"     Kendall's Tau detects whether one variable tends to increase")
    print(f"     as the other increases, but does not measure linearity.")
    print(f"     A significant τ does not imply a linear relationship.")

    print(f"\n  3. Sensitive to excessive ties.")
    print(f"     When many observations share the same rank (many tied values),")
    print(f"     Tau-b is attenuated. Interpret with caution when data is")
    print(f"     heavily discretised (e.g. Likert items with few levels).")

    print(f"\n  4. Small effect sizes at large sample sizes.")
    print(f"     In large samples, even negligible τ values may reach p < 0.05.")
    print(f"     Always evaluate effect size (τ) alongside p-value.")

    print(f"\n  5. Confounding not controlled.")
    print(f"     This analysis does not adjust for third variables.")
    print(f"     Significant associations may be explained by confounders.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(res: dict,
                 dx: dict,
                 dy: dict,
                 col_x: str,
                 col_y: str,
                 dataset_path: str) -> str:
    """Assemble the complete text report as a single string."""
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tau = res["tau"]
    p   = res["p_value"]
    n   = res["n"]
    sig = res["significant"]

    lines = []

    # Header
    lines += [
        SEPARATOR,
        "  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "  Module  : Kendall's Tau Correlation",
        f"  Dataset : {dataset_path}",
        f"  Date    : {ts}",
        SEPARATOR,
    ]

    # Descriptive Statistics
    lines += ["", "  DESCRIPTIVE STATISTICS", SEPARATOR2]
    label_w = 22; val_w = 18
    lines.append(
        f"  {'Statistic':<{label_w}} {str(col_x)[:val_w]:>{val_w}} {str(col_y)[:val_w]:>{val_w}}"
    )
    lines.append(f"  {'-'*label_w} {'-'*val_w} {'-'*val_w}")
    rows = [
        ("Sample Size (n)", "n"),
        ("Mean",            "mean"),
        ("Median",          "median"),
        ("Std Deviation",   "std"),
        ("Minimum",         "min"),
        ("Maximum",         "max"),
    ]
    for label, key in rows:
        lines.append(
            f"  {label:<{label_w}} {fmt(dx[key]):>{val_w}} {fmt(dy[key]):>{val_w}}"
        )

    # Test Results
    lines += [
        "", "  KENDALL'S TAU RESULTS", SEPARATOR2,
        f"  Variable 1                    : {col_x}",
        f"  Variable 2                    : {col_y}",
        f"  Test                          : Kendall's Tau-b (two-tailed)",
        "",
        f"  Kendall's Tau (τ)             : {tau:.4f}",
        f"  p-value (two-tailed)          : {p:.4f}",
        f"  Sample Size (n)               : {n:,}",
        f"  Direction                     : {res['direction']}",
        f"  Strength                      : {res['strength']}",
        "",
        f"  H₀ : No monotonic association",
        f"  H₁ : A monotonic association exists",
        f"  Decision : {'Reject H₀' if sig else 'Fail to Reject H₀'}",
        f"  Result   : {interpret_p(p)}",
    ]

    # Interpretation
    lines += ["", "  INTERPRETATION", SEPARATOR2]
    if sig:
        lines += [
            f"  A statistically significant monotonic association was found",
            f"  between '{col_x}' and '{col_y}'.",
            f"  (τ = {tau:.4f}, p = {p:.4f})",
        ]
        if tau > 0:
            lines.append(f"  As '{col_x}' increases, '{col_y}' tends to increase.")
        else:
            lines.append(f"  As '{col_x}' increases, '{col_y}' tends to decrease.")
    else:
        lines += [
            f"  No statistically significant monotonic association was found",
            f"  between '{col_x}' and '{col_y}'.",
            f"  (τ = {tau:.4f}, p = {p:.4f})",
        ]

    if n > 200 and sig:
        lines += [
            "",
            f"  [LARGE SAMPLE NOTE] n = {n:,}. Small τ values can be",
            f"  statistically significant at large n. Evaluate practical",
            f"  importance of τ = {tau:.4f} in the context of your research.",
        ]

    # Medical Interpretation
    lines += ["", "  MEDICAL INTERPRETATION", SEPARATOR2]
    strength_lower = res["strength"].lower()
    if sig:
        direction_word = "positive" if tau > 0 else "negative"
        lines += [
            f"  The {strength_lower} {direction_word} association (τ = {tau:.4f})",
            f"  between '{col_x}' and '{col_y}' indicates that",
        ]
        if tau > 0:
            lines.append(
                f"  higher values of '{col_x}' are consistently associated "
                f"with higher values of '{col_y}'."
            )
        else:
            lines.append(
                f"  higher values of '{col_x}' are consistently associated "
                f"with lower values of '{col_y}'."
            )
        lines += [
            "",
            f"  Statistical significance (p = {p:.4f}) confirms this is",
            f"  unlikely to be due to chance. However, statistical significance",
            f"  does not imply clinical importance. Evaluate the magnitude of",
            f"  τ = {tau:.4f} in the context of your clinical research question.",
        ]
    else:
        lines += [
            f"  No meaningful monotonic association was detected between",
            f"  '{col_x}' and '{col_y}' (τ = {tau:.4f}, p = {p:.4f}).",
        ]

    # Limitations
    lines += [
        "", "  LIMITATIONS", SEPARATOR2,
        "  • Correlation does NOT imply causation.",
        "  • Detects monotonic association only — not linearity.",
        "  • Sensitive to excessive ties in the data.",
        "  • Small effect sizes may reach significance in large samples.",
        "  • Confounding variables have not been controlled for.",
    ]

    # Footer
    lines += ["", SEPARATOR, "  END OF REPORT", SEPARATOR, ""]

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """Write the report to outputs/kendall_tau_result.txt."""
    print_section("SAVE REPORT")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n  [OK] Report saved: {os.path.abspath(REPORT_FILE)}")
    except PermissionError:
        print(f"  [ERROR] Permission denied: cannot write to {REPORT_FILE}")
    except OSError as exc:
        print(f"  [ERROR] Failed to save report: {exc}")


# ======================================================================
# MAIN ORCHESTRATOR
# ======================================================================

def run_kendall_tau_module() -> None:
    """
    Interactive entry point for the Kendall's Tau Correlation module.

    Steps:
        1. Load CSV
        2. Display columns
        3. Select two variables
        4. Clean data
        5. Descriptive statistics
        6. Run Kendall's Tau
        7. Display results
        8. Interpretation
        9. Limitations
        10. Save report
    """
    print_header()
    print("  This module computes Kendall's Tau-b correlation.")
    print("  Suitable for ordinal variables, ranked data, and small samples.")
    print("  Corrects automatically for tied ranks.")

    # Step 1: Load
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select variables
    print_section("STEP 3 | SELECT VARIABLES")
    col_x = select_column(df, "first variable (Variable 1)")
    if col_x is None:
        sys.exit(1)

    col_y = select_column(df, "second variable (Variable 2)", exclude=[col_x])
    if col_y is None:
        sys.exit(1)

    # Step 4: Clean data
    x, y = clean_pair(df, col_x, col_y)
    if x is None:
        sys.exit(1)

    # Step 5: Descriptive statistics
    dx = compute_descriptives(x, col_x)
    dy = compute_descriptives(y, col_y)
    display_descriptives(dx, dy)

    # Step 6: Run Kendall's Tau
    print_section("RUNNING KENDALL'S TAU CORRELATION")
    res = run_kendall_tau(x, y)
    if res is None:
        print("\n  [FATAL] Kendall's Tau could not be computed.")
        sys.exit(1)

    # Step 7: Display results
    display_results(res, col_x, col_y)

    # Step 8: Interpretation
    display_interpretation(res, col_x, col_y)

    # Step 9: Limitations
    display_limitations()

    # Step 10: Save report
    report = build_report(res, dx, dy, col_x, col_y, path)
    save_report(report)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Kendall's Tau — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_kendall_tau_module()