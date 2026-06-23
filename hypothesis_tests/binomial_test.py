
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from datetime import datetime


# ==============================================================================
# CONSTANTS
# ==============================================================================

DIVIDER      = "=" * 70
THIN_DIVIDER = "-" * 70
HEADER       = "VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit"


# ==============================================================================
# UTILITY — I/O AND VALIDATION
# ==============================================================================

def create_output_dir() -> None:
    """Create the outputs/ directory if it does not exist."""
    os.makedirs("outputs", exist_ok=True)


def save_report(content: str, filename: str) -> None:
    """
    Write a report string to the outputs/ directory.

    Parameters:
        content  : Full report text.
        filename : Target filename (e.g. 'binomial_test_result.txt').
    """
    create_output_dir()
    filepath = os.path.join("outputs", filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\n[✔] Report saved → {filepath}")


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file with full validation.

    Checks (in order):
        1. Extension must be .csv.
        2. File must exist on disk.
        3. File must not be empty after loading.

    Parameters:
        filepath : Path to the CSV file supplied by the user.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError        : Non-CSV extension or empty file.
        FileNotFoundError : Path does not exist.
    """
    if not filepath.lower().endswith(".csv"):
        raise ValueError(
            f"Unsupported file type: '{filepath}'. "
            "Please provide a file with a .csv extension."
        )
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: '{filepath}'. "
            "Please verify the path and try again."
        )
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(
            f"The file '{filepath}' loaded successfully but contains no data."
        )
    return df


def display_columns(df: pd.DataFrame) -> None:
    """
    Print a formatted table of all column names and their data types.

    Parameters:
        df : Loaded dataset.
    """
    print(f"\n{DIVIDER}")
    print("  AVAILABLE COLUMNS IN DATASET")
    print(DIVIDER)
    print(f"  {'#':<5}  {'Column Name':<35}  {'Data Type':<15}  {'Non-Null':>8}  {'Null':>6}")
    print(f"  {THIN_DIVIDER}")
    total = len(df)
    for i, col in enumerate(df.columns, start=1):
        dtype    = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_cnt = total - non_null
        print(f"  {i:<5}  {col:<35}  {dtype:<15}  {non_null:>8}  {null_cnt:>6}")
    print(f"\n  Total rows: {total:,}   Total columns: {len(df.columns)}")
    print(DIVIDER)


def validate_binary_column(df: pd.DataFrame, col: str) -> list:
    """
    Validate that a column exists and contains exactly 2 unique non-null values.

    Parameters:
        df  : Dataset.
        col : Column name to validate.

    Returns:
        Sorted list of the two unique category values.

    Raises:
        ValueError : Column not found, only 1 unique value, or more than 2.
    """
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop nulls before counting unique values
    unique_vals = df[col].dropna().unique()

    if len(unique_vals) == 0:
        raise ValueError(
            f"Column '{col}' contains no non-null values. "
            "Cannot perform Binomial Test on an empty column."
        )
    if len(unique_vals) == 1:
        raise ValueError(
            f"Column '{col}' has only one unique value: {unique_vals[0]}. "
            "Binomial Test requires exactly 2 categories (binary variable)."
        )
    if len(unique_vals) > 2:
        raise ValueError(
            f"Column '{col}' has {len(unique_vals)} unique values: "
            f"{sorted(unique_vals, key=str)}. "
            "Binomial Test requires exactly 2 categories. "
            "For multiple categories, use the Chi-Square Goodness-of-Fit Test."
        )

    return sorted(unique_vals, key=str)


def validate_success_category(success_input: str,
                               unique_vals: list,
                               col: str):
    """
    Validate the user-supplied success category against the column's actual values.

    Handles both numeric (0/1) and string ('male'/'female') columns.
    Attempts type coercion to match the column's dtype.

    Parameters:
        success_input : Raw string input from the user.
        unique_vals   : The two unique values from the binary column.
        col           : Column name (for error messages).

    Returns:
        The matched success value in its original type.

    Raises:
        ValueError : Input does not match either category.
    """
    # Try direct string match first
    str_vals = [str(v) for v in unique_vals]
    if success_input in str_vals:
        idx = str_vals.index(success_input)
        return unique_vals[idx]

    # Try numeric coercion (e.g. user types "1" for integer column)
    try:
        numeric_input = float(success_input)
        for val in unique_vals:
            try:
                if float(val) == numeric_input:
                    return val
            except (ValueError, TypeError):
                continue
    except ValueError:
        pass

    raise ValueError(
        f"'{success_input}' does not match any category in column '{col}'. "
        f"Valid categories are: {[str(v) for v in unique_vals]}. "
        "Please enter one of these values exactly as shown."
    )


def validate_expected_proportion(p_input: str) -> float:
    """
    Validate the expected proportion supplied by the user.

    Must be a number strictly between 0 and 1 (exclusive).

    Parameters:
        p_input : Raw string input from the user.

    Returns:
        Validated float between 0 and 1 (exclusive).

    Raises:
        ValueError : Input is not numeric, or outside (0, 1).
    """
    try:
        p0 = float(p_input)
    except ValueError:
        raise ValueError(
            f"'{p_input}' is not a valid number. "
            "Please enter the expected proportion as a decimal (e.g. 0.50)."
        )

    if p0 <= 0 or p0 >= 1:
        raise ValueError(
            f"Expected proportion must be strictly between 0 and 1 (exclusive). "
            f"You entered: {p0}. "
            "Example valid values: 0.50, 0.30, 0.75."
        )
    return p0


def handle_missing_values(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Extract a clean Series from a column, removing null values.

    Parameters:
        df  : Dataset.
        col : Column to clean.

    Returns:
        Cleaned pd.Series with no null values.
    """
    original_n = len(df)
    series     = df[col].dropna()
    dropped    = original_n - len(series)

    if dropped > 0:
        print(f"\n  [INFO] {dropped:,} row(s) with missing values removed from '{col}'.")
    else:
        print(f"\n  [INFO] No missing values found in '{col}'.")
    print(f"  [INFO] Total observations used: {len(series):,}")

    return series.reset_index(drop=True)


# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

def compute_descriptives(series: pd.Series,
                          success_val,
                          failure_val,
                          p0: float) -> dict:
    """
    Compute all descriptive counts and proportions for the binary variable.

    Parameters:
        series      : Cleaned binary Series.
        success_val : The value treated as "success".
        failure_val : The value treated as "failure".
        p0          : Expected proportion under H₀.

    Returns:
        Dictionary of counts and proportions.
    """
    n           = len(series)
    k           = int((series == success_val).sum())
    failures    = n - k
    p_obs       = k / n if n > 0 else np.nan
    p_fail_obs  = failures / n if n > 0 else np.nan

    return {
        "n"          : n,
        "k"          : k,
        "failures"   : failures,
        "p_obs"      : p_obs,
        "p_fail_obs" : p_fail_obs,
        "p0"         : p0,
        "success_val": success_val,
        "failure_val": failure_val,
    }


def format_descriptives_block(desc: dict, col: str) -> list:
    """
    Format descriptive statistics into report lines.

    Parameters:
        desc : Dict from compute_descriptives().
        col  : Outcome column name.

    Returns:
        List of formatted strings.
    """
    lines = [
        "",
        "DESCRIPTIVE STATISTICS",
        THIN_DIVIDER,
        f"  Outcome Variable        : {col}",
        f"  Success Category        : {desc['success_val']}",
        f"  Failure Category        : {desc['failure_val']}",
        "",
        f"  {'Total Observations (N)':<30} : {desc['n']:,}",
        f"  {'Success Count (k)':<30} : {desc['k']:,}",
        f"  {'Failure Count':<30} : {desc['failures']:,}",
        "",
        f"  {'Observed Proportion (p̂)':<30} : {desc['p_obs']:.6f}  ({desc['p_obs']*100:.2f}%)",
        f"  {'Observed Failure Rate':<30} : {desc['p_fail_obs']:.6f}  ({desc['p_fail_obs']*100:.2f}%)",
        f"  {'Expected Proportion (p₀)':<30} : {desc['p0']:.6f}  ({desc['p0']*100:.2f}%)",
        "",
        f"  {'Difference (p̂ − p₀)':<30} : {desc['p_obs'] - desc['p0']:+.6f}",
        "",
    ]
    return lines


# ==============================================================================
# ASSUMPTION CHECKS
# ==============================================================================

def format_assumption_block(desc: dict, col: str) -> list:
    """
    Format assumption checks for the Binomial Test into report lines.

    Parameters:
        desc : Dict from compute_descriptives().
        col  : Outcome column name.

    Returns:
        List of formatted strings.
    """
    n  = desc["n"]
    k  = desc["k"]
    p0 = desc["p0"]

    # Check normal approximation validity (informational only — test is exact)
    norm_approx_valid = (n * p0 >= 5) and (n * (1 - p0) >= 5)
    norm_note = (
        f"n×p₀ = {n*p0:.1f} ≥ 5 and n×(1−p₀) = {n*(1-p0):.1f} ≥ 5 "
        "— Normal approximation would also be valid, but exact test is used."
        if norm_approx_valid else
        f"n×p₀ = {n*p0:.1f} or n×(1−p₀) = {n*(1-p0):.1f} < 5 "
        "— Exact Binomial Test is the correct choice (approximation invalid)."
    )

    lines = [
        "",
        "ASSUMPTION CHECKS",
        THIN_DIVIDER,
        "  1. Binary Outcome Variable",
        f"     ✔ Column '{col}' contains exactly 2 categories: "
        f"'{desc['success_val']}' and '{desc['failure_val']}'.",
        "     Each observation belongs to one and only one category.",
        "",
        "  2. Independence of Observations",
        "     Assumed — each row represents a unique patient/subject.",
        "     Verify this holds for your study design.",
        "     (Repeated measures or clustered data violate this assumption.)",
        "",
        "  3. Fixed Number of Trials",
        f"     ✔ N = {n:,} observations. The number of trials is fixed and known.",
        "",
        "  4. Constant Probability Under H₀",
        f"     ✔ Expected proportion p₀ = {p0} is fixed for all trials.",
        "     Under H₀, each observation has the same probability of success.",
        "",
        "  5. Test Selection (Exact vs Approximate)",
        f"     {norm_note}",
        "     The Exact Binomial Test (Clopper-Pearson) is used regardless",
        "     of sample size — it is always valid and never approximates.",
        "",
    ]
    return lines


# ==============================================================================
# EFFECT SIZE
# ==============================================================================

def compute_effect_size(p_obs: float, p0: float) -> dict:
    """
    Compute effect size as the absolute difference between observed
    and expected proportions.

    Interpretation thresholds:
        |Δ| < 0.05 → Negligible
        |Δ| < 0.10 → Small
        |Δ| < 0.20 → Medium
        |Δ| ≥ 0.20 → Large

    Parameters:
        p_obs : Observed proportion.
        p0    : Expected proportion under H₀.

    Returns:
        Dict with keys: difference, abs_difference, label, direction.
    """
    diff     = p_obs - p0
    abs_diff = abs(diff)

    if abs_diff < 0.05:
        label = "Negligible"
    elif abs_diff < 0.10:
        label = "Small"
    elif abs_diff < 0.20:
        label = "Medium"
    else:
        label = "Large"

    direction = (
        "above expected" if diff > 0
        else "below expected" if diff < 0
        else "equal to expected"
    )

    return {
        "difference"    : diff,
        "abs_difference": abs_diff,
        "label"         : label,
        "direction"     : direction,
    }


def format_effect_size_block(effect: dict,
                              desc: dict) -> list:
    """
    Format effect size results into report lines.

    Parameters:
        effect : Dict from compute_effect_size().
        desc   : Dict from compute_descriptives().

    Returns:
        List of formatted strings.
    """
    lines = [
        "",
        "EFFECT SIZE",
        THIN_DIVIDER,
        f"  Observed Proportion (p̂)  : {desc['p_obs']:.6f}  ({desc['p_obs']*100:.2f}%)",
        f"  Expected Proportion (p₀) : {desc['p0']:.6f}  ({desc['p0']*100:.2f}%)",
        f"  Difference (p̂ − p₀)     : {effect['difference']:+.6f}  ({effect['difference']*100:+.2f}%)",
        f"  |Difference|             : {effect['abs_difference']:.6f}",
        f"  Direction                : Observed proportion is {effect['direction']}",
        "",
        f"  Effect Size Label        : {effect['label']}",
        "  Reference: |Δ| < 0.05 → Negligible | < 0.10 → Small | < 0.20 → Medium | ≥ 0.20 → Large",
        "",
    ]
    return lines


# ==============================================================================
# INTERPRETATION
# ==============================================================================

def interpret_p(p_value: float) -> str:
    """Return graded medical interpretation of a p-value."""
    if p_value < 0.001:
        return "Highly statistically significant (p < 0.001)."
    elif p_value < 0.01:
        return "Statistically significant (p < 0.01)."
    elif p_value < 0.05:
        return "Statistically significant (p < 0.05)."
    else:
        return "No statistically significant difference detected (p ≥ 0.05)."


def format_interpretation_block(desc: dict,
                                  effect: dict,
                                  p_value: float,
                                  ci_low: float,
                                  ci_high: float,
                                  col: str) -> list:
    """
    Build the medical and statistical interpretation section.

    Parameters:
        desc    : Dict from compute_descriptives().
        effect  : Dict from compute_effect_size().
        p_value : Exact Binomial p-value.
        ci_low  : Lower bound of 95% CI.
        ci_high : Upper bound of 95% CI.
        col     : Outcome column name.

    Returns:
        List of formatted strings.
    """
    significant = p_value < 0.05
    p_interp    = interpret_p(p_value)

    if significant:
        main_conclusion = (
            f"The observed proportion of '{desc['success_val']}' ({desc['p_obs']*100:.2f}%) "
            f"differs significantly from the expected proportion ({desc['p0']*100:.2f}%)."
        )
    else:
        main_conclusion = (
            f"No statistically significant difference was detected between the "
            f"observed proportion ({desc['p_obs']*100:.2f}%) and the expected "
            f"proportion ({desc['p0']*100:.2f}%)."
        )

    lines = [
        "",
        "MEDICAL & STATISTICAL INTERPRETATION",
        THIN_DIVIDER,
        "",
        f"  Overall Result      : {p_interp}",
        f"  Decision            : {'Reject H₀' if significant else 'Fail to Reject H₀'}",
        "",
        "  HYPOTHESIS:",
        f"  H₀ : The true proportion of '{desc['success_val']}' = {desc['p0']}",
        f"  H₁ : The true proportion of '{desc['success_val']}' ≠ {desc['p0']}",
        "",
        f"  Test     : Exact Binomial Test (two-sided)",
        f"  k        : {desc['k']:,}  (successes observed)",
        f"  N        : {desc['n']:,}  (total observations)",
        f"  p̂        : {desc['p_obs']:.6f}  (observed proportion)",
        f"  p₀       : {desc['p0']:.6f}  (expected proportion under H₀)",
        f"  p-value  : {p_value:.6f}",
        f"  95% CI   : [{ci_low:.6f}, {ci_high:.6f}]",
        "",
        "  MAIN CONCLUSION:",
        f"  {main_conclusion}",
        "",
        "  EFFECT SIZE:",
        f"  The difference of {effect['difference']:+.4f} ({effect['difference']*100:+.2f}%) "
        f"is classified as {effect['label'].upper()}.",
        f"  The observed proportion is {effect['direction']}.",
        "",
        "  CLINICAL RELEVANCE:",
        f"  {'A statistically significant deviation from the expected rate was found.' if significant else 'The data are consistent with the expected proportion.'}",
        f"  The 95% confidence interval [{ci_low:.4f}, {ci_high:.4f}] "
        f"{'does not include' if not (ci_low <= desc['p0'] <= ci_high) else 'includes'} "
        f"the expected value p₀ = {desc['p0']}, "
        f"which is consistent with {'rejecting' if significant else 'failing to reject'} H₀.",
        "  Clinical significance should be assessed alongside statistical significance.",
        f"  An effect size of {effect['label'].lower()} does not automatically imply",
        "  clinical importance — consider the medical context and study design.",
        "",
        "  LIMITATIONS:",
        "    • The Binomial Test tests ONE proportion against a FIXED reference.",
        "    • For comparing two proportions against each other, use Z-Test or",
        "      Chi-Square Test of Independence.",
        "    • Independence of observations is assumed but not verified by the test.",
        "    • The expected proportion p₀ should come from literature or theory,",
        "      not estimated from the same dataset being tested.",
        "    • Correlation does not imply causation.",
        "",
    ]
    return lines


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_binomial_test(df: pd.DataFrame) -> str:
    """
    Execute the full Exact Binomial Test pipeline.

    Steps:
        1. Display columns.
        2. Collect and validate user inputs.
        3. Handle missing values.
        4. Compute descriptive statistics.
        5. Check assumptions.
        6. Run scipy.stats.binomtest().
        7. Compute effect size.
        8. Assemble and save report.

    Parameters:
        df : Loaded pandas DataFrame.

    Returns:
        Full report string.
    """
    # ------------------------------------------------------------------ #
    # STEP 1 — Display columns
    # ------------------------------------------------------------------ #
    display_columns(df)

    # ------------------------------------------------------------------ #
    # STEP 2 — Collect and validate user inputs
    # ------------------------------------------------------------------ #
    print(f"\n{THIN_DIVIDER}")
    print("  EXACT BINOMIAL TEST — INPUT SELECTION")
    print(THIN_DIVIDER)
    print("  SUPPORT2 examples:")
    print("    Binary column : death  (0 = alive, 1 = dead)")
    print("    Binary column : sex    (male / female)")
    print()

    # --- Select binary outcome column ---
    print("  Step 1: Select the BINARY OUTCOME column (exactly 2 categories).")
    outcome_col = input("  Enter column name: ").strip()
    unique_vals = validate_binary_column(df, outcome_col)
    print(f"\n  [INFO] Categories detected: {[str(v) for v in unique_vals]}")

    # --- Select success category ---
    print(f"\n  Step 2: Select the SUCCESS CATEGORY.")
    print(f"  Categories available: {[str(v) for v in unique_vals]}")
    success_input = input("  Enter success category: ").strip()
    success_val   = validate_success_category(success_input, unique_vals, outcome_col)
    failure_val   = [v for v in unique_vals if v != success_val][0]
    print(f"\n  [INFO] Success → '{success_val}'   |   Failure → '{failure_val}'")

    # --- Get expected proportion ---
    print(f"\n  Step 3: Enter the EXPECTED PROPORTION (p₀) under H₀.")
    print("  This is the reference rate from literature or clinical guidelines.")
    print("  Example: 0.50 (50%) for equal distribution, 0.30 (30%) for a benchmark rate.")
    p0_input = input("  Enter expected proportion (0 < p₀ < 1): ").strip()
    p0 = validate_expected_proportion(p0_input)
    print(f"\n  [INFO] Expected proportion under H₀: {p0} ({p0*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # STEP 3 — Handle missing values
    # ------------------------------------------------------------------ #
    print(f"\n{THIN_DIVIDER}")
    print("  DATA CLEANING")
    print(THIN_DIVIDER)
    series = handle_missing_values(df, outcome_col)

    # Re-validate after dropping nulls (edge case: all nulls in one category)
    post_clean_vals = series.unique()
    if len(post_clean_vals) < 2:
        raise ValueError(
            f"After removing missing values, column '{outcome_col}' contains "
            f"only one category: {post_clean_vals}. "
            "Cannot perform Binomial Test."
        )

    # ------------------------------------------------------------------ #
    # STEP 4 — Descriptive statistics
    # ------------------------------------------------------------------ #
    desc = compute_descriptives(series, success_val, failure_val, p0)

    if desc["n"] == 0:
        raise ValueError("No valid observations remain after cleaning. Cannot run test.")

    # Display group summary
    print(f"\n{THIN_DIVIDER}")
    print("  OUTCOME SUMMARY")
    print(THIN_DIVIDER)
    print(f"  {'Category':<30}  {'Count':>8}  {'Proportion':>12}  {'%':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*12}  {'-'*8}")
    print(f"  {str(success_val):<30}  {desc['k']:>8,}  {desc['p_obs']:>12.6f}  {desc['p_obs']*100:>7.2f}%")
    print(f"  {str(failure_val):<30}  {desc['failures']:>8,}  {desc['p_fail_obs']:>12.6f}  {desc['p_fail_obs']*100:>7.2f}%")
    print(f"  {'TOTAL':<30}  {desc['n']:>8,}  {'1.000000':>12}  {'100.00%':>8}")
    print(f"\n  Expected proportion (p₀) : {p0} ({p0*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # STEP 5 — Run Exact Binomial Test
    # ------------------------------------------------------------------ #
    print(f"\n  [INFO] Running Exact Binomial Test ...")
    result = binomtest(k=desc["k"], n=desc["n"], p=p0, alternative="two-sided")

    p_value   = result.pvalue
    p_hat     = result.proportion_estimate   # = k/n

    # 95% Exact Clopper-Pearson CI
    ci = result.proportion_ci(confidence_level=0.95, method="exact")
    ci_low  = float(ci.low)
    ci_high = float(ci.high)

    # Guard against NaN (should not occur with valid inputs, but defensive)
    if math.isnan(p_value):
        raise ValueError(
            "Binomial Test returned NaN p-value. "
            "Please check your inputs (k, n, p₀)."
        )

    print(f"  [INFO] Test complete. p-value = {p_value:.6f}")

    # ------------------------------------------------------------------ #
    # STEP 6 — Effect size
    # ------------------------------------------------------------------ #
    effect = compute_effect_size(desc["p_obs"], p0)

    # ------------------------------------------------------------------ #
    # STEP 7 — Assemble report
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_header = [
        DIVIDER,
        HEADER,
        DIVIDER,
        f"  Test              : Exact Binomial Test (two-sided)",
        f"  Date / Time       : {timestamp}",
        f"  Outcome Variable  : {outcome_col}",
        f"  Success Category  : {success_val}",
        f"  Failure Category  : {failure_val}",
        f"  Expected Prop (p₀): {p0} ({p0*100:.1f}%)",
        f"  Total N           : {desc['n']:,}",
        DIVIDER,
    ]

    test_results_block = [
        "",
        "BINOMIAL TEST RESULTS",
        THIN_DIVIDER,
        f"  Test Applied              : Exact Binomial Test (two-sided)",
        f"  Successes (k)             : {desc['k']:,}",
        f"  Trials (N)                : {desc['n']:,}",
        f"  Observed Proportion (p̂)  : {p_hat:.6f}  ({p_hat*100:.2f}%)",
        f"  Expected Proportion (p₀) : {p0:.6f}  ({p0*100:.2f}%)",
        f"  P-Value (two-sided)       : {p_value:.6f}",
        f"  95% CI (Clopper-Pearson)  : [{ci_low:.6f}, {ci_high:.6f}]",
        f"  Statistical Significance  : {interpret_p(p_value)}",
        f"  Decision                  : {'Reject H₀' if p_value < 0.05 else 'Fail to Reject H₀'}",
        "",
    ]

    # Assemble all sections
    report_lines = (
        report_header
        + format_descriptives_block(desc, outcome_col)
        + format_assumption_block(desc, outcome_col)
        + test_results_block
        + format_effect_size_block(effect, desc)
        + format_interpretation_block(desc, effect, p_value, ci_low, ci_high, outcome_col)
        + [DIVIDER, "END OF REPORT", DIVIDER]
    )

    report = "\n".join(report_lines)

    # Print and save
    print("\n" + report)
    save_report(report, "binomial_test_result.txt")

    return report


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Interactive entry point for the Binomial Test module.

    Guides the user through:
        1. CSV file path input.
        2. Binary column and success category selection.
        3. Expected proportion entry.
        4. Full Exact Binomial Test analysis.
    """
    print(f"\n{DIVIDER}")
    print(f"  VM-WIT-STATS | VM Medical College")
    print(f"  Module: Exact Binomial Test")
    print(DIVIDER)

    # --- Load CSV ---
    csv_path = input("\n  Enter the full path to your CSV file: ").strip()

    try:
        df = load_csv(csv_path)
        print(f"\n  [✔] Dataset loaded successfully.  Shape: {df.shape}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    # --- Run Binomial Test ---
    try:
        run_binomial_test(df)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    print("\n  [✔] Binomial Test analysis complete.")


if __name__ == "__main__":
    main()