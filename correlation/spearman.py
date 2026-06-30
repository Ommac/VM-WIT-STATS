"""
======================================================================
VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
======================================================================
Module      : Spearman Rank Correlation
File        : correlation/spearman.py
Description : Measures the strength and direction of a monotonic
              association between two variables.

              Uses scipy.stats.spearmanr().

WHEN TO USE:
    - Ordinal variables
    - Ranked data
    - Continuous variables that violate normality
    - Monotonic (not necessarily linear) relationships

WHEN NOT TO USE:
    - You need to measure a strictly linear relationship (use Pearson)
    - One variable is binary (use Point-Biserial)
    - Both variables are nominal/categorical (use Chi-Square or Cramér's V)

ASSUMPTIONS:
    1. Two variables measured on at least an ordinal scale.
    2. Observations are independent.
    3. The relationship is assumed to be approximately monotonic under H1.

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
REPORT_FILE = os.path.join(REPORT_DIR, "spearman_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Spearman Rank Correlation
{SEPARATOR}
"""

# Suggested interpretation thresholds for |ρ|.
STRENGTH_THRESHOLDS = [
    (0.20, "Very Weak"),
    (0.40, "Weak"),
    (0.60, "Moderate"),
    (0.80, "Strong"),
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


def fmt_p(p: float) -> str:
    """Format a p-value: 'p < 0.001' or 'p = X.XXXX'."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def interpret_strength(rho: float) -> str:
    """
    Return the strength label for a given |ρ| value.

    Suggested interpretation of Spearman correlation magnitude
    (not a universal standard):
        |ρ| < 0.20  → Very Weak
        0.20–0.39   → Weak
        0.40–0.59   → Moderate
        0.60–0.79   → Strong
        ≥ 0.80      → Very Strong
    """
    abs_rho = abs(rho)
    for threshold, label in STRENGTH_THRESHOLDS:
        if abs_rho < threshold:
            return label
    return "Very Strong"


def interpret_direction(rho: float) -> str:
    """Return 'Positive', 'Negative', or 'None' for a correlation value."""
    if abs(rho) < 1e-10:
        return "None (no correlation)"
    return "Positive" if rho > 0 else "Negative"


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
    print("\n  NOTE: Spearman Rank Correlation requires two numeric or")
    print("        ordinal variables. Each row must represent one")
    print("        independent observation.")

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
    str : Column name, or None if a valid selection cannot be made.
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

        coerced = pd.to_numeric(df[col], errors="coerce")
        valid_n = int(coerced.notna().sum())
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

    valid = np.isfinite(x_raw) & np.isfinite(y_raw)
    dropped = int((~valid).sum())
    x = x_raw[valid]
    y = y_raw[valid]
    n = len(x)

    if dropped > 0:
        print(f"  Dropped (missing/infinite): {dropped:,} row(s)")
    print(f"  Pairs used for analysis    : {n:,}")

    if n < 3:
        print(f"\n  [ERROR] Only {n} valid pair(s) remain. "
              f"Spearman Rank Correlation requires at least 3 observations.")
        return None, None

    if float(x.max()) - float(x.min()) < 1e-10:
        print(f"\n  [ERROR] '{col_x}' is constant (all values identical). "
              f"Spearman correlation is undefined for a constant variable.")
        return None, None
    if float(y.max()) - float(y.min()) < 1e-10:
        print(f"\n  [ERROR] '{col_y}' is constant (all values identical). "
              f"Spearman correlation is undefined for a constant variable.")
        return None, None

    if np.array_equal(x, y):
        print(f"\n  [WARNING] The two selected columns contain identical values. "
              f"ρ will be 1.0 by definition. Verify your column selection.")

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
# SPEARMAN RANK CORRELATION
# ======================================================================

def run_spearman(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    Compute Spearman Rank Correlation using scipy.stats.spearmanr().

    Parameters
    ----------
    x, y : clean float64 arrays of equal length

    Returns
    -------
    dict with keys: rho, p_value, n, direction, strength, significant
    Returns None if scipy returns NaN for either the coefficient or
    the p-value (e.g. n=2, or a constant variable that slipped through).
    """
    n = len(x)

    try:
        result = stats.spearmanr(x, y)
    except Exception as exc:
        print(f"  [ERROR] scipy.stats.spearmanr failed: {exc}")
        return None

    rho     = float(result.correlation)
    p_value = float(result.pvalue)

    if math.isnan(rho) or math.isnan(p_value):
        print("  [ERROR] Spearman Rank Correlation returned NaN. "
              "This may occur with very small samples or when one "
              "variable has zero variance.")
        return None

    # Clamp to [-1, 1] — guard against floating-point overshoot
    rho = max(-1.0, min(1.0, rho))

    direction   = interpret_direction(rho)
    strength    = interpret_strength(rho)
    significant = bool(p_value < 0.05)   # bool() ensures Python bool, never numpy.bool_

    return {
        "rho"        : rho,
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
    """Print Spearman Rank Correlation results in a formatted table."""
    print_section("SPEARMAN RANK CORRELATION RESULTS")

    rho = res["rho"]
    p   = res["p_value"]
    n   = res["n"]
    sig = res["significant"]

    print(f"\n  Variable 1       : {col_x}")
    print(f"  Variable 2       : {col_y}")
    print(f"  Test             : Spearman Rank Correlation (two-tailed)")
    print(f"  Null Hypothesis  : No monotonic association exists")
    print(f"  Alt. Hypothesis  : A monotonic association exists")
    print(f"  Significance (α) : 0.05\n")

    rho_label = "Spearman Correlation Coefficient (ρ)"
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {rho_label:<40} {rho:>20.4f}")
    print(f"  {'p-value':<40} {fmt_p(p):>20}")
    print(f"  {'Sample Size (n)':<40} {n:>20,}")
    print(f"  {'Direction':<40} {res['direction']:>20}")
    print(f"  {'Strength':<40} {res['strength']:>20}")
    print(f"  {SEPARATOR2[:68]}")

    decision = "Reject H₀" if sig else "Fail to Reject H₀"
    print(f"\n  Decision : {decision}")
    print(f"  Result   : {interpret_p(p)}")


# ======================================================================
# ASSUMPTION CHECKS
# ======================================================================

def display_assumptions(res: dict, col_x: str, col_y: str) -> None:
    """Display assumption checks relevant to Spearman Rank Correlation."""
    print_section("ASSUMPTION CHECKS")

    n = res["n"]

    print(f"\n  1. Variable Scale")
    print(f"     CONFIRMED: '{col_x}' and '{col_y}' are ordinal or continuous.")

    print(f"\n  2. Independence of Observations")
    print(f"     ASSUMED: Each row represents one independent subject.")

    print(f"\n  3. Monotonic Relationship")
    print(f"     ASSUMED under H₁: Spearman detects monotonic associations,")
    print(f"     not necessarily linear ones. The relationship between")
    print(f"     '{col_x}' and '{col_y}' should be approximately monotonic")
    print(f"     for this correlation to be meaningfully interpreted.")

    print(f"\n  4. Sample Size")
    if n > 200:
        print(f"     [NOTE] Large sample (n = {n:,}). Small correlations may")
        print(f"     reach statistical significance. Evaluate effect size (ρ)")
        print(f"     alongside the p-value.")
    elif n < 30:
        print(f"     [NOTE] Small sample (n = {n:,}). Interpret results with caution.")
    else:
        print(f"     Sample size (n = {n:,}) is adequate for this analysis.")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(res: dict, col_x: str, col_y: str) -> None:
    """Display statistical and medical interpretation of the result."""
    print_section("INTERPRETATION")

    rho = res["rho"]
    p   = res["p_value"]
    sig = res["significant"]
    n   = res["n"]

    print(f"\n  STATISTICAL INTERPRETATION:")
    if sig:
        print(f"  A statistically significant monotonic association was found")
        print(f"  between '{col_x}' and '{col_y}'.")
        print(f"  (ρ = {rho:.4f}, {fmt_p(p)})")
        if rho > 0:
            print(f"  As '{col_x}' increases, '{col_y}' tends to increase.")
        else:
            print(f"  As '{col_x}' increases, '{col_y}' tends to decrease.")
    else:
        print(f"  No statistically significant monotonic association was found")
        print(f"  between '{col_x}' and '{col_y}'.")
        print(f"  (ρ = {rho:.4f}, {fmt_p(p)})")

    print(f"\n  Association direction : {res['direction']}")
    print(f"  Association strength  : {res['strength']}")

    if n > 200 and sig:
        print(f"\n  [LARGE SAMPLE NOTE] n = {n:,}.")
        print(f"  With large samples, even small ρ values can reach statistical")
        print(f"  significance. Evaluate the practical importance of ρ = {rho:.4f}")
        print(f"  in the context of your research question.")

    print(f"\n  MEDICAL INTERPRETATION:")
    strength_lower = res["strength"].lower()
    if sig:
        direction_word = "positive" if rho > 0 else "negative"
        print(f"  The {strength_lower} {direction_word} association (ρ = {rho:.4f})")
        print(f"  between '{col_x}' and '{col_y}'")
        if rho > 0:
            print(f"  indicates that higher values of '{col_x}' are consistently")
            print(f"  associated with higher values of '{col_y}'.")
        else:
            print(f"  indicates that higher values of '{col_x}' are consistently")
            print(f"  associated with lower values of '{col_y}'.")
        print(f"\n  Statistical significance ({fmt_p(p)}) confirms this association")
        print(f"  is unlikely to be due to chance alone. However, statistical")
        print(f"  significance does not imply clinical importance. Evaluate the")
        print(f"  magnitude of ρ = {rho:.4f} in the context of your specific")
        print(f"  clinical or research question.")
    else:
        print(f"  No meaningful monotonic association was detected between")
        print(f"  '{col_x}' and '{col_y}' (ρ = {rho:.4f}, {fmt_p(p)}).")
        print(f"  The data do not support a consistent directional relationship")
        print(f"  between these two variables in this sample.")


def display_limitations() -> None:
    """Display limitations relevant to Spearman Rank Correlation."""
    print_section("LIMITATIONS")

    print(f"\n  1. Correlation does NOT imply causation.")
    print(f"     An association between variables does not establish that")
    print(f"     one causes the other.")

    print(f"\n  2. Detects monotonic association only.")
    print(f"     Spearman measures whether one variable tends to increase or")
    print(f"     decrease as the other does, but does not imply linearity.")

    print(f"\n  3. Sensitive to excessive tied ranks.")
    print(f"     When many observations share the same rank, ρ may be")
    print(f"     attenuated. Interpret with caution for heavily discretised data.")

    print(f"\n  4. Small effect sizes at large sample sizes.")
    print(f"     In large samples, even negligible ρ values may reach p < 0.05.")
    print(f"     Always evaluate effect size (ρ) alongside p-value.")


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
    rho = res["rho"]
    p   = res["p_value"]
    n   = res["n"]
    sig = res["significant"]

    lines = []

    # Header
    lines += [
        SEPARATOR,
        "  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "  Module  : Spearman Rank Correlation",
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
        "", "  SPEARMAN RANK CORRELATION RESULTS", SEPARATOR2,
        f"  Variable 1                    : {col_x}",
        f"  Variable 2                    : {col_y}",
        f"  Test                          : Spearman Rank Correlation (two-tailed)",
        "",
        f"  Spearman Correlation Coefficient (ρ) : {rho:.4f}",
        f"  p-value                              : {fmt_p(p)}",
        f"  Sample Size (n)                      : {n:,}",
        f"  Direction                            : {res['direction']}",
        f"  Strength                             : {res['strength']}",
        "",
        f"  H₀ : No monotonic association exists",
        f"  H₁ : A monotonic association exists",
        f"  Decision : {'Reject H₀' if sig else 'Fail to Reject H₀'}",
        f"  Result   : {interpret_p(p)}",
    ]

    # Assumption Checks
    lines += ["", "  ASSUMPTION CHECKS", SEPARATOR2]
    lines.append(f"  1. Variable Scale       : CONFIRMED — ordinal or continuous")
    lines.append(f"  2. Independence         : ASSUMED")
    lines.append(f"  3. Monotonic Relationship: ASSUMED under H₁")
    if n > 200:
        lines.append(f"  4. Sample Size          : Large (n = {n:,}) — evaluate effect size")
    elif n < 30:
        lines.append(f"  4. Sample Size          : Small (n = {n:,}) — interpret with caution")
    else:
        lines.append(f"  4. Sample Size          : Adequate (n = {n:,})")

    # Interpretation
    lines += ["", "  INTERPRETATION", SEPARATOR2]
    if sig:
        lines += [
            f"  A statistically significant monotonic association was found",
            f"  between '{col_x}' and '{col_y}'.",
            f"  (ρ = {rho:.4f}, {fmt_p(p)})",
        ]
        if rho > 0:
            lines.append(f"  As '{col_x}' increases, '{col_y}' tends to increase.")
        else:
            lines.append(f"  As '{col_x}' increases, '{col_y}' tends to decrease.")
    else:
        lines += [
            f"  No statistically significant monotonic association was found",
            f"  between '{col_x}' and '{col_y}'.",
            f"  (ρ = {rho:.4f}, {fmt_p(p)})",
        ]

    if n > 200 and sig:
        lines += [
            "",
            f"  [LARGE SAMPLE NOTE] n = {n:,}. Small ρ values can be",
            f"  statistically significant at large n. Evaluate practical",
            f"  importance of ρ = {rho:.4f} in the context of your research.",
        ]

    # Medical Interpretation
    lines += ["", "  MEDICAL INTERPRETATION", SEPARATOR2]
    strength_lower = res["strength"].lower()
    if sig:
        direction_word = "positive" if rho > 0 else "negative"
        lines += [
            f"  The {strength_lower} {direction_word} association (ρ = {rho:.4f})",
            f"  between '{col_x}' and '{col_y}' indicates that",
        ]
        if rho > 0:
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
            f"  Statistical significance ({fmt_p(p)}) confirms this is",
            f"  unlikely to be due to chance. However, statistical significance",
            f"  does not imply clinical importance. Evaluate the magnitude of",
            f"  ρ = {rho:.4f} in the context of your clinical research question.",
        ]
    else:
        lines += [
            f"  No meaningful monotonic association was detected between",
            f"  '{col_x}' and '{col_y}' (ρ = {rho:.4f}, {fmt_p(p)}).",
        ]

    # Limitations
    lines += [
        "", "  LIMITATIONS", SEPARATOR2,
        "  • Correlation does NOT imply causation.",
        "  • Detects monotonic association only — not linearity.",
        "  • Sensitive to excessive tied ranks.",
        "  • Small effect sizes may reach significance in large samples.",
    ]

    # Footer
    lines += ["", SEPARATOR, "  END OF REPORT", SEPARATOR, ""]

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """Write the report to outputs/spearman_result.txt."""
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

def run_spearman_module() -> None:
    """
    Interactive entry point for the Spearman Rank Correlation module.

    Steps:
        1. Load CSV
        2. Display columns
        3. Select two variables
        4. Clean data
        5. Descriptive statistics
        6. Run Spearman Rank Correlation
        7. Display results
        8. Assumption checks
        9. Interpretation
        10. Limitations
        11. Save report
    """
    print_header()
    print("  This module computes Spearman Rank Correlation.")
    print("  Measures monotonic association between two ordinal or")
    print("  continuous variables. Does not assume linearity or normality.")

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

    # Step 6: Run Spearman Rank Correlation
    print_section("RUNNING SPEARMAN RANK CORRELATION")
    res = run_spearman(x, y)
    if res is None:
        print("\n  [FATAL] Spearman Rank Correlation could not be computed.")
        sys.exit(1)

    # Step 7: Display results
    display_results(res, col_x, col_y)

    # Step 8: Assumption checks
    display_assumptions(res, col_x, col_y)

    # Step 9: Interpretation
    display_interpretation(res, col_x, col_y)

    # Step 10: Limitations
    display_limitations()

    # Step 11: Save report
    report = build_report(res, dx, dy, col_x, col_y, path)
    save_report(report)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Spearman Rank Correlation — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_spearman_module()