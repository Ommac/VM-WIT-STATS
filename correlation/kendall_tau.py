"""
kendall_tau.py
==============
VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit

Kendall's Tau Correlation Module

Measures the strength and direction of monotonic association between two
variables using concordant and discordant pairs.

Appropriate for:
    - Ordinal variables and ranked data
    - Small samples
    - Data with many tied ranks
    - Non-normal distributions

Statistical notes:
    - scipy.stats.kendalltau() computes Tau-b by default, which corrects
      for ties. This is the appropriate version for ordinal/medical data
      and is consistent with SAS, SPSS, and R (method="kendall").
    - Confidence intervals are not included: no closed-form exact CI exists
      for Kendall's Tau. Bootstrap CIs are not standard in publication-grade
      Kendall's Tau reporting and exceed the scope of this module.

Author: VM-WIT-STATS Toolkit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "kendall_tau_result.txt")


# ==============================================================================
# UTILITY
# ==============================================================================

def _separator(char: str = "=", width: int = 72) -> str:
    return char * width


def _section(title: str) -> str:
    sep = _separator()
    return f"\n{sep}\n{title}\n{sep}\n"


def _dual_print(lines: list, file_handle) -> None:
    """Print each line to console and write to file simultaneously."""
    for line in lines:
        print(line)
        file_handle.write(line + "\n")


def _format_p(p: float) -> str:
    """Format p-value for publication reporting."""
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


# ==============================================================================
# FILE LOADING
# ==============================================================================

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file with comprehensive error handling.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None on failure.
    """
    if not filepath or not isinstance(filepath, str):
        print("[ERROR] Invalid file path provided.")
        return None

    filepath = filepath.strip()

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
    except PermissionError:
        print(f"[ERROR] Permission denied: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("[ERROR] The CSV file is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"[ERROR] CSV parse error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error loading file: {e}")
        return None

    if df.empty:
        print("[ERROR] Dataset is empty after loading.")
        return None

    print(f"\n[INFO] File loaded successfully: {filepath}")
    print(f"       Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return df


# ==============================================================================
# COLUMN DISPLAY
# ==============================================================================

def display_columns(df: pd.DataFrame) -> None:
    """Display available columns with dtype and missing value count."""
    print("\n" + _separator("-"))
    print("AVAILABLE COLUMNS")
    print(_separator("-"))
    for i, col in enumerate(df.columns, 1):
        n_miss = df[col].isna().sum()
        print(f"  [{i:>2}] {col:<30} dtype={df[col].dtype}  missing={n_miss}")
    print(_separator("-"))


# ==============================================================================
# VARIABLE SELECTION
# ==============================================================================

def _resolve_column(token: str, cols: list) -> str:
    """Resolve a column name or 1-based index string to a column name."""
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(cols):
            return cols[idx]
        return None
    if token in cols:
        return token
    return None


def select_variable(df: pd.DataFrame, prompt: str, exclude: str = None) -> str:
    """
    Interactively select a variable column by name or number.

    Parameters:
        df (pd.DataFrame): Loaded dataset.
        prompt (str): Input prompt shown to the user.
        exclude (str): Column name to exclude from valid choices.

    Returns:
        str: Selected column name, or None on failure.
    """
    cols = list(df.columns)
    while True:
        try:
            raw = input(prompt).strip()
            if not raw:
                print("[ERROR] No input provided.")
                continue
            col = _resolve_column(raw, cols)
            if col is None:
                print(f"[ERROR] Column '{raw}' not found.")
                continue
            if exclude and col == exclude:
                print(f"[ERROR] '{col}' is already selected as the other variable. "
                      "Choose a different column.")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[ERROR] '{col}' is not numeric. Both variables must be numeric or ordinal.")
                continue
            print(f"[INFO] Selected: {col}")
            return col
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_data(df: pd.DataFrame, var1: str, var2: str) -> bool:
    """
    Validate the selected variables and dataset for Kendall's Tau analysis.

    Checks:
        - Columns exist in the dataset
        - Variables are not identical
        - Variables are numeric
        - Missing values (warning, not fatal)
        - Constant variables (fatal)
        - Minimum sample size after cleaning

    Parameters:
        df (pd.DataFrame): Loaded dataset.
        var1 (str): First variable column name.
        var2 (str): Second variable column name.

    Returns:
        bool: True if validation passes, False on fatal error.
    """
    # Column existence
    for var in [var1, var2]:
        if var not in df.columns:
            print(f"[ERROR] Column '{var}' not found in dataset.")
            return False

    # Identical variable check
    if var1 == var2:
        print("[ERROR] Both variables are the same column. "
              "Select two different columns.")
        return False

    # Numeric check
    for var in [var1, var2]:
        if not pd.api.types.is_numeric_dtype(df[var]):
            print(f"[ERROR] Variable '{var}' is not numeric.")
            return False

    # Missing values — warning only
    for var in [var1, var2]:
        n_miss = df[var].isna().sum()
        if n_miss > 0:
            pct = 100 * n_miss / len(df)
            print(f"[WARNING] '{var}' has {n_miss} missing value(s) ({pct:.1f}%). "
                  "These rows will be excluded from analysis.")

    # Clean subset for further checks
    subset = df[[var1, var2]].dropna()
    n = len(subset)

    if n < 3:
        print(f"[ERROR] Insufficient observations after removing missing values: n = {n}. "
              "Minimum required: 3.")
        return False
    if n < 10:
        print(f"[WARNING SEVERE] Very small sample size: n = {n}. "
              "Results are highly unreliable.")
    elif n < 20:
        print(f"[WARNING] Small sample size: n = {n}. Interpret results with caution.")
    elif n < 30:
        print(f"[CAUTION] Sample size: n = {n}. Results may be unstable.")

    # Constant variable check
    for var in [var1, var2]:
        if subset[var].nunique() <= 1:
            print(f"[ERROR] Variable '{var}' is constant (no variance). "
                  "Kendall's Tau cannot be computed.")
            return False

    print(f"[INFO] Validation passed. Usable observations: {n}")
    return True


# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

def descriptive_statistics(df: pd.DataFrame, var1: str, var2: str) -> pd.DataFrame:
    """
    Compute basic descriptive statistics for both variables.

    Reports N, Mean, Median, SD, Min, Max, and missing count — the
    standard descriptive summary appropriate for Kendall's Tau reporting.

    Parameters:
        df (pd.DataFrame): Cleaned dataset (missing values already removed).
        var1 (str): First variable name.
        var2 (str): Second variable name.

    Returns:
        pd.DataFrame: Descriptive statistics table.
    """
    rows = []
    for var in [var1, var2]:
        s = df[var]
        rows.append({
            "Variable": var,
            "N": len(s),
            "Mean": s.mean(),
            "Median": s.median(),
            "SD": s.std(ddof=1),
            "Min": s.min(),
            "Max": s.max(),
            "Missing (original)": 0,  # filled after merge with original counts
        })
    return pd.DataFrame(rows)


# ==============================================================================
# KENDALL'S TAU COMPUTATION
# ==============================================================================

def compute_kendall_tau(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute Kendall's Tau-b correlation coefficient and p-value.

    Uses scipy.stats.kendalltau(), which computes Tau-b — the standard
    version that corrects for tied ranks. Tau-b is the appropriate choice
    for ordinal and ranked medical data and is consistent with SAS PROC
    CORR, SPSS, and R cor(..., method='kendall').

    Tau-b formula:
        τ_b = (C - D) / sqrt((C + D + T_x)(C + D + T_y))
    where C = concordant pairs, D = discordant pairs,
    T_x = ties in x only, T_y = ties in y only.

    The p-value is computed using the asymptotic normal approximation,
    which is appropriate for n ≥ 10. For very small n, exact p-values
    are preferable but are not provided by scipy at this time.

    Parameters:
        x (np.ndarray): First variable values.
        y (np.ndarray): Second variable values.

    Returns:
        dict: tau, p_value, n, concordant/discordant pair counts.
    """
    try:
        tau, p_value = stats.kendalltau(x, y)

        # Concordant and discordant pair counts
        n = len(x)
        n_pairs = n * (n - 1) // 2
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                prod = dx * dy
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1

        return {
            "tau": float(tau),
            "p_value": float(p_value),
            "n": n,
            "n_pairs": n_pairs,
            "concordant": concordant,
            "discordant": discordant,
            "ties": n_pairs - concordant - discordant,
        }
    except Exception as e:
        print(f"[ERROR] Kendall's Tau computation failed: {e}")
        return None


# ==============================================================================
# INTERPRETATION
# ==============================================================================

def interpret_tau(tau: float, p_value: float, n: int,
                  var1: str, var2: str) -> dict:
    """
    Interpret the Kendall's Tau result statistically and clinically.

    Strength thresholds (adapted for Tau from Botsch, 2011 and
    Dancey & Reidy, 2011 — widely used in medical statistics):
        |τ| < 0.10  : Negligible
        0.10–0.29   : Weak
        0.30–0.49   : Moderate
        0.50–0.69   : Strong
        ≥ 0.70      : Very Strong

    Note: Tau values are systematically lower than Pearson r or Spearman ρ
    for the same dataset (typically τ ≈ 0.67 × ρ). Strength labels account
    for this by using lower thresholds than those applied to r or ρ.

    Parameters:
        tau (float): Kendall's Tau-b coefficient.
        p_value (float): Two-tailed p-value.
        n (int): Sample size.
        var1 (str): Name of first variable.
        var2 (str): Name of second variable.

    Returns:
        dict: direction, strength, decision, statistical_interp,
              medical_interp strings.
    """
    abs_tau = abs(tau)

    # Direction
    if abs_tau < 0.001:
        direction = "No association"
    elif tau > 0:
        direction = "Positive (concordant)"
    else:
        direction = "Negative (discordant)"

    # Strength
    if abs_tau < 0.10:
        strength = "Negligible"
    elif abs_tau < 0.30:
        strength = "Weak"
    elif abs_tau < 0.50:
        strength = "Moderate"
    elif abs_tau < 0.70:
        strength = "Strong"
    else:
        strength = "Very Strong"

    # Hypothesis decision
    alpha = 0.05
    if p_value < alpha:
        decision = f"Reject H₀ (p = {_format_p(p_value)} < {alpha})"
        sig_statement = "statistically significant"
    else:
        decision = f"Fail to Reject H₀ (p = {_format_p(p_value)} ≥ {alpha})"
        sig_statement = "not statistically significant"

    # Statistical interpretation
    if tau >= 0:
        direction_text = (
            f"As {var1} increases, {var2} tends to increase (concordant pairs dominate)."
        )
    else:
        direction_text = (
            f"As {var1} increases, {var2} tends to decrease (discordant pairs dominate)."
        )

    statistical_interp = (
        f"Kendall's Tau-b = {tau:.4f} indicates a {strength.lower()} "
        f"{direction.lower()} monotonic association between {var1} and {var2}. "
        f"The result is {sig_statement} at α = {alpha}. "
        f"{direction_text}"
    )

    # Medical interpretation
    if p_value >= alpha:
        medical_interp = (
            f"There is insufficient evidence of a monotonic association between "
            f"{var1} and {var2} in this sample (τ = {tau:.4f}, p = {_format_p(p_value)}). "
            "This may reflect a true absence of association, insufficient sample size, "
            "or high variability in the data. Clinical conclusions should not be drawn "
            "from a non-significant result alone."
        )
    else:
        if abs_tau < 0.10:
            practical = (
                "Although statistically significant, the association is negligible in magnitude. "
                "This may reflect a very large sample detecting a trivially small effect. "
                "Clinical relevance is unlikely."
            )
        elif abs_tau < 0.30:
            practical = (
                "The association is weak. While statistically significant, the clinical "
                "utility of this relationship is limited. Consider whether this difference "
                "is meaningful in the clinical context."
            )
        elif abs_tau < 0.50:
            practical = (
                "The association is moderate and may be of clinical relevance. "
                "Interpret in the context of the specific variables and clinical setting."
            )
        elif abs_tau < 0.70:
            practical = (
                "The association is strong and likely clinically meaningful. "
                "Higher values of one variable are consistently associated with "
                "higher (or lower) values of the other in the majority of pairs."
            )
        else:
            practical = (
                "The association is very strong and highly consistent across pairs. "
                "This level of monotonic concordance is clinically notable."
            )

        medical_interp = (
            f"A {strength.lower()} {direction.lower().replace('(concordant)', '').replace('(discordant)', '').strip()} "
            f"monotonic association was found between {var1} and {var2} "
            f"(τ = {tau:.4f}, p = {_format_p(p_value)}, n = {n}). "
            f"{practical}"
        )

    return {
        "direction": direction,
        "strength": strength,
        "decision": decision,
        "statistical_interp": statistical_interp,
        "medical_interp": medical_interp,
    }


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(
    var1: str,
    var2: str,
    n_original: int,
    missing_var1: int,
    missing_var2: int,
    desc: pd.DataFrame,
    result: dict,
    interp: dict,
    file_handle,
) -> None:
    """
    Write the complete Kendall's Tau report to console and file simultaneously.

    Parameters:
        var1: First variable name.
        var2: Second variable name.
        n_original: Row count before removing missing values.
        missing_var1: Missing count for var1 in original dataset.
        missing_var2: Missing count for var2 in original dataset.
        desc: Descriptive statistics DataFrame.
        result: Output of compute_kendall_tau().
        interp: Output of interpret_tau().
        file_handle: Open file handle for writing.
    """
    out = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit")
    out.append("KENDALL'S TAU CORRELATION ANALYSIS")
    out.append(_separator())
    out.append(f"Variable 1        : {var1}")
    out.append(f"Variable 2        : {var2}")
    out.append(f"Original rows     : {n_original}")
    out.append(f"Missing ({var1})  : {missing_var1}")
    out.append(f"Missing ({var2})  : {missing_var2}")
    out.append(f"Analysed (n)      : {result['n']}")
    out.append(f"Method            : Kendall's Tau-b (ties-corrected)")
    _dual_print(out, file_handle); out = []

    # ── DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────
    out.append(_section("1. DESCRIPTIVE STATISTICS"))
    out.append(
        f"  {'Variable':<28} {'N':>6} {'Mean':>10} {'Median':>10} "
        f"{'SD':>10} {'Min':>10} {'Max':>10}"
    )
    out.append("  " + _separator("-", 86))
    for _, row in desc.iterrows():
        out.append(
            f"  {str(row['Variable']):<28} {int(row['N']):>6} "
            f"{row['Mean']:>10.4f} {row['Median']:>10.4f} "
            f"{row['SD']:>10.4f} {row['Min']:>10.4f} {row['Max']:>10.4f}"
        )
    _dual_print(out, file_handle); out = []

    # ── CORRELATION RESULT ────────────────────────────────────────────────────
    out.append(_section("2. KENDALL'S TAU CORRELATION"))
    out.append(f"  Kendall's Tau-b (τ) : {result['tau']:>10.4f}")
    out.append(f"  p-value             : {_format_p(result['p_value']):>10}")
    out.append(f"  Sample Size (n)     : {result['n']:>10}")
    out.append(f"  Total Pairs         : {result['n_pairs']:>10}")
    out.append(f"  Concordant Pairs    : {result['concordant']:>10}")
    out.append(f"  Discordant Pairs    : {result['discordant']:>10}")
    out.append(f"  Tied Pairs          : {result['ties']:>10}")
    out.append("")
    out.append(f"  Direction : {interp['direction']}")
    out.append(f"  Strength  : {interp['strength']}")
    _dual_print(out, file_handle); out = []

    # ── HYPOTHESIS TEST ────────────────────────────────────────────────────────
    out.append(_section("3. HYPOTHESIS TEST"))
    out.append("  H₀: No monotonic association between the two variables (τ = 0).")
    out.append("  H₁: A monotonic association exists (τ ≠ 0).")
    out.append(f"  Significance level (α) : 0.05")
    out.append(f"  Decision               : {interp['decision']}")
    _dual_print(out, file_handle); out = []

    # ── INTERPRETATION ────────────────────────────────────────────────────────
    out.append(_section("4. INTERPRETATION"))
    out.append("  Strength Reference (Kendall's Tau-b):")
    out.append("    |τ| < 0.10  : Negligible")
    out.append("    0.10–0.29   : Weak")
    out.append("    0.30–0.49   : Moderate")
    out.append("    0.50–0.69   : Strong")
    out.append("    ≥ 0.70      : Very Strong")
    out.append("")
    out.append("  Statistical Interpretation:")
    out.append(f"    {interp['statistical_interp']}")
    out.append("")
    out.append("  Medical / Clinical Interpretation:")
    out.append(f"    {interp['medical_interp']}")
    _dual_print(out, file_handle); out = []

    # ── LIMITATIONS ───────────────────────────────────────────────────────────
    out.append(_section("5. LIMITATIONS"))
    out.append("""
  1. CORRELATION IS NOT CAUSATION
     A significant Kendall's Tau indicates a monotonic association only.
     It does not establish causal direction or mechanisms.

  2. MONOTONIC ASSOCIATION ONLY
     Kendall's Tau detects whether one variable consistently increases (or
     decreases) as the other increases. It will not detect non-monotonic
     relationships (e.g., U-shaped associations).

  3. TIES SENSITIVITY
     Although Tau-b corrects for tied ranks, a very high proportion of ties
     reduces statistical power and the interpretability of the coefficient.
     Verify that the ordinal scale has sufficient resolution for the data.

  4. LINEARITY NOT IMPLIED
     A significant Tau does not imply that the association is linear.
     Kendall's Tau is a rank-based measure and makes no linearity assumption.

  5. LARGE-SAMPLE INFLATION
     In large samples (n > 200), even negligibly small values of τ may
     reach statistical significance. Always evaluate effect magnitude
     alongside the p-value.
    """)
    _dual_print(out, file_handle); out = []

    # ── FOOTER ────────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("END OF REPORT — VM-WIT-STATS | VM Medical College")
    out.append("For academic publication, report τ, p-value, and sample size.")
    out.append("This output is research-grade and intended for trained professionals.")
    out.append(_separator())
    _dual_print(out, file_handle)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Main entry point for Kendall's Tau correlation analysis.

    Workflow:
        1. Prompt for CSV file path
        2. Load dataset
        3. Display available columns
        4. User selects two variables
        5. Validate data
        6. Remove missing values
        7. Compute Kendall's Tau-b
        8. Interpret results
        9. Display and save report
    """
    print(_separator())
    print("VM-WIT-STATS | VM Medical College")
    print("KENDALL'S TAU CORRELATION ANALYSIS")
    print(_separator())

    # ── STEP 1: File path ─────────────────────────────────────────────────────
    try:
        filepath = input("\nEnter the full path to your CSV file: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Cancelled.")
        sys.exit(0)

    # ── STEP 2: Load ──────────────────────────────────────────────────────────
    df = load_csv(filepath)
    if df is None:
        print("[FATAL] Could not load data. Exiting.")
        sys.exit(1)

    # ── STEP 3: Display columns ───────────────────────────────────────────────
    display_columns(df)

    # ── STEP 4: Select variables ──────────────────────────────────────────────
    var1 = select_variable(
        df,
        prompt="\nEnter column name or number for VARIABLE 1: ",
    )
    if var1 is None:
        print("[FATAL] No variable selected. Exiting.")
        sys.exit(1)

    var2 = select_variable(
        df,
        prompt="\nEnter column name or number for VARIABLE 2: ",
        exclude=var1,
    )
    if var2 is None:
        print("[FATAL] No variable selected. Exiting.")
        sys.exit(1)

    # ── STEP 5: Validate ──────────────────────────────────────────────────────
    missing_var1 = int(df[var1].isna().sum())
    missing_var2 = int(df[var2].isna().sum())
    n_original = len(df)

    if not validate_data(df, var1, var2):
        print("[FATAL] Validation failed. Exiting.")
        sys.exit(1)

    # ── STEP 6: Clean ─────────────────────────────────────────────────────────
    clean = df[[var1, var2]].dropna().reset_index(drop=True)

    # ── STEP 7: Descriptive statistics ────────────────────────────────────────
    desc = descriptive_statistics(clean, var1, var2)

    # ── STEP 8: Compute Kendall's Tau ─────────────────────────────────────────
    print("\n[INFO] Computing Kendall's Tau-b...")
    result = compute_kendall_tau(clean[var1].values, clean[var2].values)
    if result is None:
        print("[FATAL] Correlation computation failed. Exiting.")
        sys.exit(1)

    # ── STEP 9: Interpret ─────────────────────────────────────────────────────
    interp = interpret_tau(result["tau"], result["p_value"], result["n"], var1, var2)

    # ── STEP 10: Report ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            generate_report(
                var1=var1,
                var2=var2,
                n_original=n_original,
                missing_var1=missing_var1,
                missing_var2=missing_var2,
                desc=desc,
                result=result,
                interp=interp,
                file_handle=fh,
            )
        print(f"\n[INFO] Report saved to: {OUTPUT_FILE}")
    except OSError as e:
        print(f"[ERROR] Could not save report: {e}")
        print("[INFO] Displaying report to console only.")
        import io
        generate_report(
            var1=var1,
            var2=var2,
            n_original=n_original,
            missing_var1=missing_var1,
            missing_var2=missing_var2,
            desc=desc,
            result=result,
            interp=interp,
            file_handle=io.StringIO(),
        )


if __name__ == "__main__":
    main()