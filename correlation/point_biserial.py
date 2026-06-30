"""
======================================================================
VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
======================================================================
Module      : Point-Biserial Correlation
File        : correlation/point_biserial.py
Description : Measures the strength and direction of the relationship
              between one binary variable and one continuous variable.

              Uses scipy.stats.pointbiserialr(), which is mathematically
              equivalent to Pearson correlation when the binary variable
              is coded as 0/1.

WHEN TO USE:
    - One binary variable (e.g. sex, smoking status, survival outcome)
    - One continuous variable (e.g. blood pressure, biomarker level)

WHEN NOT TO USE:
    - The grouping variable has more than two categories (use Eta or ANOVA)
    - Both variables are continuous (use Pearson or Spearman)

ASSUMPTIONS:
    1. The binary variable has exactly two categories.
    2. The continuous variable is measured on an interval/ratio scale.
    3. Observations are independent.

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
REPORT_FILE = os.path.join(REPORT_DIR, "point_biserial_result.txt")

HEADER = f"""
{SEPARATOR}
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Point-Biserial Correlation
{SEPARATOR}
"""

# Suggested interpretation thresholds for |r_pb|.
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


def interpret_strength(r: float) -> str:
    """
    Return the strength label for a given |r_pb| value.

    Suggested interpretation of Point-Biserial Correlation magnitude:
        |r| < 0.20  → Very Weak
        0.20–0.39   → Weak
        0.40–0.59   → Moderate
        0.60–0.79   → Strong
        ≥ 0.80      → Very Strong
    """
    abs_r = abs(r)
    for threshold, label in STRENGTH_THRESHOLDS:
        if abs_r < threshold:
            return label
    return "Very Strong"


def interpret_direction(r: float) -> str:
    """Return 'Positive', 'Negative', or 'None' for a correlation value."""
    if abs(r) < 1e-10:
        return "None (no correlation)"
    return "Positive" if r > 0 else "Negative"


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
    print("\n  NOTE: Point-Biserial Correlation requires one binary variable")
    print("        and one continuous numeric variable.")
    print("        Example: sex (Male/Female) and systolic_bp")

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

def select_binary_column(df: pd.DataFrame) -> tuple[str, dict] | tuple[None, None]:
    """
    Prompt user to select the binary variable.

    Validates that the column contains exactly two unique non-missing
    values. Accepts any two-category encoding (0/1, Yes/No, True/False,
    Male/Female, Alive/Dead, etc.) and automatically encodes the values
    to 0 and 1 in sorted order. The encoding mapping is displayed.

    Returns
    -------
    (col_name, encoding_map) or (None, None) on failure.
    encoding_map maps original label -> 0 or 1.
    """
    print_section("STEP 3 | SELECT BINARY VARIABLE")
    print("\n  Select the BINARY variable (must have exactly 2 categories).")
    print("  Examples: sex (Male/Female), smoking (Yes/No), outcome (0/1)")

    while True:
        col = input("\n  Enter column name for binary variable: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue

        unique_vals = df[col].dropna().unique()
        if len(unique_vals) != 2:
            print(f"  [ERROR] Column '{col}' has {len(unique_vals)} unique value(s): "
                  f"{list(unique_vals)}. Point-Biserial Correlation requires "
                  f"exactly 2 categories.")
            continue

        # Sort for deterministic encoding: first (alphabetically/numerically) → 0
        sorted_vals = sorted(unique_vals, key=str)
        encoding = {sorted_vals[0]: 0, sorted_vals[1]: 1}

        print(f"\n  [OK] '{col}' selected as binary variable.")
        print(f"  Encoding applied:")
        print(f"    '{sorted_vals[0]}' → 0")
        print(f"    '{sorted_vals[1]}' → 1")

        return col, encoding


def select_continuous_column(df: pd.DataFrame,
                              exclude: str) -> str | None:
    """
    Prompt user to select the continuous numeric variable.

    Parameters
    ----------
    df      : DataFrame
    exclude : Column name already selected as the binary variable.

    Returns
    -------
    str : Column name, or None on repeated failure (caller handles loop).
    """
    print_section("STEP 4 | SELECT CONTINUOUS VARIABLE")
    print("\n  Select the CONTINUOUS variable (numeric, e.g. systolic_bp, age).")

    while True:
        col = input("\n  Enter column name for continuous variable: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue
        if col == exclude:
            print(f"  [ERROR] '{col}' is already selected as the binary variable. "
                  f"Choose a different column.")
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
               col_bin: str,
               col_cont: str,
               encoding: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extract, encode, and clean the binary and continuous columns.

    Steps:
        1. Apply the 0/1 encoding to the binary column.
        2. Coerce the continuous column to float64.
        3. Drop rows where either value is missing or ±Inf.
        4. Enforce minimum sample size (n ≥ 3).
        5. Verify both binary groups are still present after cleaning.
        6. Check the continuous variable is not constant.

    Returns
    -------
    (x_bin, y_cont) : tuple of clean float64 arrays, or (None, None) on failure.
    """
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    x_raw = df[col_bin].map(encoding)
    y_raw = pd.to_numeric(df[col_cont], errors="coerce")

    original_n = len(df)
    print(f"\n  Original rows: {original_n:,}")

    x_arr = x_raw.to_numpy(dtype=float)
    y_arr = y_raw.to_numpy(dtype=float)

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    dropped = int((~valid).sum())
    x = x_arr[valid]
    y = y_arr[valid]
    n = len(x)

    if dropped > 0:
        print(f"  Dropped (missing/infinite): {dropped:,} row(s)")
    print(f"  Pairs used for analysis    : {n:,}")

    if n < 3:
        print(f"\n  [ERROR] Only {n} valid pair(s) remain. "
              f"Point-Biserial Correlation requires at least 3 observations.")
        return None, None

    # Verify both groups still present
    unique_after = np.unique(x)
    if len(unique_after) != 2:
        print(f"\n  [ERROR] After removing missing values, only "
              f"{len(unique_after)} binary group(s) remain. "
              f"Both groups must have at least one valid observation.")
        return None, None

    # Constant continuous variable
    if float(y.max()) - float(y.min()) < 1e-10:
        print(f"\n  [ERROR] '{col_cont}' is constant (all values identical). "
              f"Point-Biserial Correlation is undefined for a constant "
              f"continuous variable.")
        return None, None

    print(f"\n  [OK] Data validated and ready for analysis.")
    return x, y


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_descriptives(x_bin: np.ndarray,
                          y_cont: np.ndarray,
                          col_bin: str,
                          col_cont: str,
                          encoding: dict) -> dict:
    """
    Compute descriptive statistics for both variables and group means.

    Returns
    -------
    dict with keys:
        n, group_labels (dict 0/1 -> original label), group_sizes,
        group_means (mean of continuous var within each group),
        cont_mean, cont_median, cont_std, cont_min, cont_max
    """
    n = len(x_bin)

    # Reverse encoding: 0/1 -> original label
    reverse_enc = {v: k for k, v in encoding.items()}

    group_sizes = {
        reverse_enc[0]: int(np.sum(x_bin == 0)),
        reverse_enc[1]: int(np.sum(x_bin == 1)),
    }
    group_means = {
        reverse_enc[0]: float(y_cont[x_bin == 0].mean()) if group_sizes[reverse_enc[0]] > 0 else float("nan"),
        reverse_enc[1]: float(y_cont[x_bin == 1].mean()) if group_sizes[reverse_enc[1]] > 0 else float("nan"),
    }

    return {
        "n"           : n,
        "group_labels": reverse_enc,
        "group_sizes" : group_sizes,
        "group_means" : group_means,
        "cont_mean"   : float(np.mean(y_cont)),
        "cont_median" : float(np.median(y_cont)),
        "cont_std"    : float(np.std(y_cont, ddof=1)) if n > 1 else float("nan"),
        "cont_min"    : float(np.min(y_cont)),
        "cont_max"    : float(np.max(y_cont)),
    }


def display_descriptives(desc: dict, col_bin: str, col_cont: str) -> None:
    """Print descriptive statistics for both variables and group breakdown."""
    print_section("DESCRIPTIVE STATISTICS")

    print(f"\n  Overall Sample Size (n) : {desc['n']:,}")

    print(f"\n  BINARY VARIABLE: {col_bin}")
    print(f"  {'-'*50}")
    for label, count in desc["group_sizes"].items():
        print(f"    {str(label):<25} n = {count:,}")

    print(f"\n  CONTINUOUS VARIABLE: {col_cont}")
    print(f"  {'-'*50}")
    print(f"    Mean              : {fmt(desc['cont_mean'])}")
    print(f"    Median            : {fmt(desc['cont_median'])}")
    print(f"    Standard Deviation: {fmt(desc['cont_std'])}")
    print(f"    Minimum           : {fmt(desc['cont_min'])}")
    print(f"    Maximum           : {fmt(desc['cont_max'])}")

    print(f"\n  MEAN OF '{col_cont}' BY GROUP:")
    print(f"  {'-'*50}")
    for label, mean_val in desc["group_means"].items():
        print(f"    {str(label):<25} mean = {fmt(mean_val)}")


# ======================================================================
# POINT-BISERIAL CORRELATION
# ======================================================================

def run_point_biserial(x_bin: np.ndarray, y_cont: np.ndarray) -> dict | None:
    """
    Compute Point-Biserial Correlation using scipy.stats.pointbiserialr().

    Parameters
    ----------
    x_bin  : clean float64 array of 0/1 values
    y_cont : clean float64 array of continuous values (same length)

    Returns
    -------
    dict with keys: r, p_value, n, direction, strength, significant
    Returns None if scipy returns NaN.
    """
    n = len(x_bin)

    try:
        result = stats.pointbiserialr(x_bin, y_cont)
    except Exception as exc:
        print(f"  [ERROR] scipy.stats.pointbiserialr failed: {exc}")
        return None

    r       = float(result.correlation)
    p_value = float(result.pvalue)

    if math.isnan(r) or math.isnan(p_value):
        print("  [ERROR] Point-Biserial Correlation returned NaN. "
              "This may occur when the continuous variable has zero "
              "variance within a group or overall.")
        return None

    # Clamp to [-1, 1] — guard against floating-point overshoot
    r = max(-1.0, min(1.0, r))

    direction   = interpret_direction(r)
    strength    = interpret_strength(r)
    significant = bool(p_value < 0.05)   # bool() ensures Python bool, never numpy.bool_

    return {
        "r"          : r,
        "p_value"    : p_value,
        "n"          : n,
        "direction"  : direction,
        "strength"   : strength,
        "significant": significant,
    }


# ======================================================================
# DISPLAY TEST RESULTS
# ======================================================================

def display_results(res: dict, col_bin: str, col_cont: str) -> None:
    """Print Point-Biserial Correlation results in a formatted table."""
    print_section("POINT-BISERIAL RESULTS")

    r   = res["r"]
    p   = res["p_value"]
    n   = res["n"]
    sig = res["significant"]

    print(f"\n  Binary Variable     : {col_bin}")
    print(f"  Continuous Variable : {col_cont}")
    print(f"  Test                : Point-Biserial Correlation")
    print(f"  Null Hypothesis     : No association between the variables")
    print(f"  Alt. Hypothesis     : An association exists")
    print(f"  Significance (α)    : 0.05\n")

    r_label = "Point-Biserial Correlation (r_pb)"
    print(f"  {SEPARATOR2[:68]}")
    print(f"  {r_label:<40} {r:>20.4f}")
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

def check_outliers_iqr(y_cont: np.ndarray) -> dict:
    """
    Simple IQR-based outlier check on the continuous variable.

    Returns
    -------
    dict with keys: n_outliers, lower_bound, upper_bound
    """
    q1 = float(np.percentile(y_cont, 25))
    q3 = float(np.percentile(y_cont, 75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = int(np.sum((y_cont < lower) | (y_cont > upper)))
    return {
        "n_outliers" : n_outliers,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def display_assumptions(res: dict, outlier_info: dict,
                         col_bin: str, col_cont: str) -> None:
    """Display assumption checks relevant to Point-Biserial Correlation."""
    print_section("ASSUMPTION CHECKS")

    n = res["n"]

    print(f"\n  1. Binary Variable")
    print(f"     CONFIRMED: '{col_bin}' has exactly 2 categories.")

    print(f"\n  2. Continuous Variable")
    print(f"     CONFIRMED: '{col_cont}' is numeric.")

    print(f"\n  3. Independence of Observations")
    print(f"     ASSUMED: Each row represents one independent subject.")

    print(f"\n  4. Outlier Check (IQR rule) — '{col_cont}'")
    if outlier_info["n_outliers"] > 0:
        print(f"     [WARNING] {outlier_info['n_outliers']:,} potential outlier(s) "
              f"detected (outside [{outlier_info['lower_bound']:.4f}, "
              f"{outlier_info['upper_bound']:.4f}]).")
        print(f"     Point-Biserial Correlation is sensitive to outliers in the "
              f"continuous variable. Consider reviewing these values.")
    else:
        print(f"     No outliers detected using the 1.5×IQR rule.")

    print(f"\n  5. Sample Size")
    if n > 200:
        print(f"     [NOTE] Large sample (n = {n:,}). Small correlations may "
              f"reach statistical significance. Evaluate effect size (r_pb) "
              f"alongside the p-value.")
    elif n < 30:
        print(f"     [NOTE] Small sample (n = {n:,}). Interpret results with caution.")
    else:
        print(f"     Sample size (n = {n:,}) is adequate for this analysis.")


# ======================================================================
# INTERPRETATION
# ======================================================================

def display_interpretation(res: dict, col_bin: str, col_cont: str) -> None:
    """Display statistical and medical interpretation of the result."""
    print_section("INTERPRETATION")

    r   = res["r"]
    p   = res["p_value"]
    sig = res["significant"]
    n   = res["n"]

    print(f"\n  STATISTICAL INTERPRETATION:")
    if sig:
        print(f"  A statistically significant association was found between")
        print(f"  '{col_bin}' and '{col_cont}'.")
        print(f"  (r_pb = {r:.4f}, {fmt_p(p)})")
    else:
        print(f"  No statistically significant association was found between")
        print(f"  '{col_bin}' and '{col_cont}'.")
        print(f"  (r_pb = {r:.4f}, {fmt_p(p)})")

    print(f"\n  Direction : {res['direction']}")
    print(f"  Strength  : {res['strength']}")

    if n > 200 and sig:
        print(f"\n  [LARGE SAMPLE NOTE] n = {n:,}.")
        print(f"  With large samples, even small correlations can reach statistical")
        print(f"  significance. Evaluate the practical importance of r_pb = {r:.4f}")
        print(f"  in the context of your research question.")

    print(f"\n  MEDICAL INTERPRETATION:")
    strength_lower = res["strength"].lower()
    if sig:
        direction_word = "positive" if r > 0 else "negative"
        print(f"  The {strength_lower} {direction_word} correlation (r_pb = {r:.4f})")
        print(f"  between '{col_bin}' and '{col_cont}' indicates a measurable")
        print(f"  association between group membership and the outcome.")
        print(f"\n  Statistical significance ({fmt_p(p)}) confirms this association")
        print(f"  is unlikely to be due to chance alone. However, statistical")
        print(f"  significance does not imply clinical importance. Evaluate the")
        print(f"  magnitude of r_pb = {r:.4f} in the context of your specific")
        print(f"  clinical or research question.")
    else:
        print(f"  No meaningful association was detected between '{col_bin}' and")
        print(f"  '{col_cont}' (r_pb = {r:.4f}, {fmt_p(p)}).")
        print(f"  The data do not support a consistent relationship between")
        print(f"  these two variables in this sample.")


def display_limitations() -> None:
    """Display limitations relevant to Point-Biserial Correlation."""
    print_section("LIMITATIONS")

    print(f"\n  1. Correlation does NOT imply causation.")
    print(f"     An association between variables does not establish that")
    print(f"     one causes the other.")

    print(f"\n  2. Only one binary and one continuous variable are supported.")
    print(f"     This analysis does not account for additional variables or")
    print(f"     confounders.")

    print(f"\n  3. Sensitive to outliers in the continuous variable.")
    print(f"     Extreme values can disproportionately influence r_pb.")

    print(f"\n  4. Binary variable coding affects only direction, not magnitude.")
    print(f"     Reversing the 0/1 encoding flips the sign of r_pb but does")
    print(f"     not change its absolute value.")

    print(f"\n  5. Small effect sizes at large sample sizes.")
    print(f"     In large samples, even negligible correlations may reach")
    print(f"     p < 0.05. Always evaluate effect size (r_pb) alongside p-value.")


# ======================================================================
# REPORT GENERATION
# ======================================================================

def build_report(res: dict,
                 desc: dict,
                 outlier_info: dict,
                 col_bin: str,
                 col_cont: str,
                 encoding: dict,
                 dataset_path: str) -> str:
    """Assemble the complete text report as a single string."""
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r   = res["r"]
    p   = res["p_value"]
    n   = res["n"]
    sig = res["significant"]

    lines = []

    # Header
    lines += [
        SEPARATOR,
        "  VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "  Module  : Point-Biserial Correlation",
        f"  Dataset : {dataset_path}",
        f"  Date    : {ts}",
        SEPARATOR,
    ]

    # Encoding note
    reverse_enc = desc["group_labels"]
    lines += [
        "",
        "  BINARY VARIABLE ENCODING",
        SEPARATOR2,
        f"  '{reverse_enc[0]}' → 0",
        f"  '{reverse_enc[1]}' → 1",
    ]

    # Descriptive Statistics
    lines += ["", "  DESCRIPTIVE STATISTICS", SEPARATOR2]
    lines.append(f"  Overall Sample Size (n) : {desc['n']:,}")
    lines.append("")
    lines.append(f"  Binary Variable: {col_bin}")
    for label, count in desc["group_sizes"].items():
        lines.append(f"    {str(label):<25} n = {count:,}")
    lines.append("")
    lines.append(f"  Continuous Variable: {col_cont}")
    lines.append(f"    Mean              : {fmt(desc['cont_mean'])}")
    lines.append(f"    Median            : {fmt(desc['cont_median'])}")
    lines.append(f"    Standard Deviation: {fmt(desc['cont_std'])}")
    lines.append(f"    Minimum           : {fmt(desc['cont_min'])}")
    lines.append(f"    Maximum           : {fmt(desc['cont_max'])}")
    lines.append("")
    lines.append(f"  Mean of '{col_cont}' by Group:")
    for label, mean_val in desc["group_means"].items():
        lines.append(f"    {str(label):<25} mean = {fmt(mean_val)}")

    # Test Results
    lines += [
        "", "  POINT-BISERIAL RESULTS", SEPARATOR2,
        f"  Binary Variable                : {col_bin}",
        f"  Continuous Variable            : {col_cont}",
        f"  Test                           : Point-Biserial Correlation",
        "",
        f"  Point-Biserial Correlation (r_pb) : {r:.4f}",
        f"  p-value                           : {fmt_p(p)}",
        f"  Sample Size (n)                   : {n:,}",
        f"  Direction                         : {res['direction']}",
        f"  Strength                          : {res['strength']}",
        "",
        f"  H₀ : No association between the binary and continuous variable",
        f"  H₁ : An association exists",
        f"  Decision : {'Reject H₀' if sig else 'Fail to Reject H₀'}",
        f"  Result   : {interpret_p(p)}",
    ]

    # Assumption Checks
    lines += ["", "  ASSUMPTION CHECKS", SEPARATOR2]
    lines.append(f"  1. Binary Variable     : CONFIRMED — '{col_bin}' has 2 categories")
    lines.append(f"  2. Continuous Variable : CONFIRMED — '{col_cont}' is numeric")
    lines.append(f"  3. Independence        : ASSUMED")
    if outlier_info["n_outliers"] > 0:
        lines.append(
            f"  4. Outliers (IQR rule) : {outlier_info['n_outliers']:,} potential "
            f"outlier(s) detected"
        )
    else:
        lines.append(f"  4. Outliers (IQR rule) : None detected")
    if n > 200:
        lines.append(f"  5. Sample Size         : Large (n = {n:,}) — evaluate effect size")
    elif n < 30:
        lines.append(f"  5. Sample Size         : Small (n = {n:,}) — interpret with caution")
    else:
        lines.append(f"  5. Sample Size         : Adequate (n = {n:,})")

    # Interpretation
    lines += ["", "  INTERPRETATION", SEPARATOR2]
    if sig:
        lines += [
            f"  A statistically significant association was found between",
            f"  '{col_bin}' and '{col_cont}'.",
            f"  (r_pb = {r:.4f}, {fmt_p(p)})",
        ]
    else:
        lines += [
            f"  No statistically significant association was found between",
            f"  '{col_bin}' and '{col_cont}'.",
            f"  (r_pb = {r:.4f}, {fmt_p(p)})",
        ]

    if n > 200 and sig:
        lines += [
            "",
            f"  [LARGE SAMPLE NOTE] n = {n:,}. Small correlations can be",
            f"  statistically significant at large n. Evaluate practical",
            f"  importance of r_pb = {r:.4f} in the context of your research.",
        ]

    # Medical Interpretation
    lines += ["", "  MEDICAL INTERPRETATION", SEPARATOR2]
    strength_lower = res["strength"].lower()
    if sig:
        direction_word = "positive" if r > 0 else "negative"
        lines += [
            f"  The {strength_lower} {direction_word} correlation (r_pb = {r:.4f})",
            f"  between '{col_bin}' and '{col_cont}' indicates a measurable",
            f"  association between group membership and the outcome.",
            "",
            f"  Statistical significance ({fmt_p(p)}) confirms this is unlikely",
            f"  to be due to chance. However, statistical significance does not",
            f"  imply clinical importance. Evaluate the magnitude of r_pb = {r:.4f}",
            f"  in the context of your clinical research question.",
        ]
    else:
        lines += [
            f"  No meaningful association was detected between '{col_bin}' and",
            f"  '{col_cont}' (r_pb = {r:.4f}, {fmt_p(p)}).",
        ]

    # Limitations
    lines += [
        "", "  LIMITATIONS", SEPARATOR2,
        "  • Correlation does NOT imply causation.",
        "  • Only one binary and one continuous variable are supported.",
        "  • Sensitive to outliers in the continuous variable.",
        "  • Binary variable coding affects only direction, not magnitude.",
        "  • Small effect sizes may reach significance in large samples.",
    ]

    # Footer
    lines += ["", SEPARATOR, "  END OF REPORT", SEPARATOR, ""]

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """Write the report to outputs/point_biserial_result.txt."""
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

def run_point_biserial_module() -> None:
    """
    Interactive entry point for the Point-Biserial Correlation module.

    Steps:
        1. Load CSV
        2. Display columns
        3. Select binary variable (auto-encode to 0/1)
        4. Select continuous variable
        5. Clean data
        6. Descriptive statistics
        7. Run Point-Biserial Correlation
        8. Display results
        9. Assumption checks
        10. Interpretation
        11. Limitations
        12. Save report
    """
    print_header()
    print("  This module computes Point-Biserial Correlation.")
    print("  Measures the relationship between a binary variable and a")
    print("  continuous variable. Mathematically equivalent to Pearson")
    print("  correlation when the binary variable is coded as 0/1.")

    # Step 1: Load
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Display columns
    display_columns(df)

    # Step 3: Select binary variable
    col_bin, encoding = select_binary_column(df)
    if col_bin is None:
        sys.exit(1)

    # Step 4: Select continuous variable
    col_cont = select_continuous_column(df, exclude=col_bin)
    if col_cont is None:
        sys.exit(1)

    # Step 5: Clean data
    x_bin, y_cont = clean_pair(df, col_bin, col_cont, encoding)
    if x_bin is None:
        sys.exit(1)

    # Step 6: Descriptive statistics
    desc = compute_descriptives(x_bin, y_cont, col_bin, col_cont, encoding)
    display_descriptives(desc, col_bin, col_cont)

    # Step 7: Run Point-Biserial Correlation
    print_section("RUNNING POINT-BISERIAL CORRELATION")
    res = run_point_biserial(x_bin, y_cont)
    if res is None:
        print("\n  [FATAL] Point-Biserial Correlation could not be computed.")
        sys.exit(1)

    # Step 8: Display results
    display_results(res, col_bin, col_cont)

    # Step 9: Assumption checks
    outlier_info = check_outliers_iqr(y_cont)
    display_assumptions(res, outlier_info, col_bin, col_cont)

    # Step 10: Interpretation
    display_interpretation(res, col_bin, col_cont)

    # Step 11: Limitations
    display_limitations()

    # Step 12: Save report
    report = build_report(res, desc, outlier_info, col_bin, col_cont,
                           encoding, path)
    save_report(report)

    print(f"\n{SEPARATOR}")
    print(f"  VM-WIT-STATS | Point-Biserial Correlation — Analysis Complete")
    print(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_point_biserial_module()