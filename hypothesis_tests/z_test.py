
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
from datetime import datetime



def create_output_dir():
    """Ensure the outputs/ directory exists."""
    os.makedirs("outputs", exist_ok=True)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with full validation.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not filepath.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported. Please provide a .csv file.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    return df


def display_columns(df: pd.DataFrame):
    """Display all column names with data types."""
    print("\n" + "=" * 60)
    print("AVAILABLE COLUMNS IN DATASET")
    print("=" * 60)
    print(f"{'#':<5} {'Column Name':<35} {'Data Type'}")
    print("-" * 60)
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), start=1):
        print(f"{i:<5} {col:<35} {dtype}")
    print(f"\nTotal rows: {len(df)} | Total columns: {len(df.columns)}")
    print("=" * 60)


def validate_numeric_column(df: pd.DataFrame, col: str):
    """Raise an error if the selected column is not numeric."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' is not numeric (dtype: {df[col].dtype}). "
            "Please select a continuous numeric column."
        )


def validate_binary_column(df: pd.DataFrame, col: str) -> list:
    """
    Validate that a column has exactly 2 unique non-null values.

    Parameters:
        df (pd.DataFrame): Dataframe.
        col (str): Column name.

    Returns:
        list: The two unique values.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(
            f"Column '{col}' has {len(unique_vals)} unique value(s): {list(unique_vals)}. "
            "Z-Test for proportions requires exactly 2 unique values (binary column)."
        )
    return sorted(unique_vals)


def handle_missing_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Drop rows with missing values in selected columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        cols (list): Columns to check.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    original_n = len(df)
    df_clean = df[cols].dropna()
    dropped = original_n - len(df_clean)
    if dropped > 0:
        print(f"\n[INFO] Missing values removed: {dropped} row(s) dropped.")
        print(f"[INFO] Rows used for analysis: {len(df_clean)}")
    else:
        print(f"\n[INFO] No missing values found. All {len(df_clean)} rows used.")
    return df_clean


def interpret_p(p_value: float) -> str:
    """Return standard medical interpretation of the p-value."""
    if p_value < 0.001:
        return "Highly statistically significant (p < 0.001)."
    elif p_value < 0.01:
        return "Statistically significant (p < 0.01)."
    elif p_value < 0.05:
        return "Statistically significant (p < 0.05)."
    else:
        return "No statistically significant difference detected (p ≥ 0.05)."


def save_report(content: str, filename: str):
    """Save report to outputs/ directory."""
    create_output_dir()
    filepath = os.path.join("outputs", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[✔] Report saved to: {filepath}")


# ==============================================================================
# ONE-SAMPLE Z-TEST (Mean vs Known Population Mean)
# ==============================================================================

def run_one_sample_z_test(df: pd.DataFrame) -> str:
    """
    Execute One-Sample Z-Test.

    Tests whether the sample mean of a continuous variable significantly
    differs from a known reference/population mean.

    Workflow:
        1. User selects continuous column.
        2. User provides known reference population mean (mu_0).
        3. User optionally provides known population std dev (sigma).
           If not provided, sample std dev is used (valid for n >= 30).
        4. Check n >= 30 assumption.
        5. Compute Z-statistic = (x_bar - mu_0) / (sigma / sqrt(n)).
        6. Compute two-tailed p-value from standard normal distribution.
        7. Compute 95% CI for sample mean.
        8. Display and save report.

    Returns:
        str: Full formatted report.
    """
    print("\n" + "=" * 60)
    print("ONE-SAMPLE Z-TEST")
    print("=" * 60)
    print("PURPOSE: Compare a sample mean against a KNOWN population")
    print("         reference mean from literature or clinical standards.")
    print("=" * 60)

    display_columns(df)

    # --- Select continuous column ---
    print("\nStep 1: Select the CONTINUOUS column to test.")
    print("        SUPPORT2 examples: 'meanbp', 'hrt', 'crea', 'age'")
    col = input("Enter column name: ").strip()
    validate_numeric_column(df, col)

    # --- Handle missing values ---
    df_clean = handle_missing_values(df, [col])
    data = df_clean[col]
    n = len(data)

    # --- Check n >= 30 ---
    if n < 30:
        print(f"\n[WARNING] n = {n}. Z-Test is most valid for n >= 30.")
        print("          Consider using T-Test for small samples.")
        cont = input("Continue anyway? (yes/no): ").strip().lower()
        if cont != "yes":
            raise ValueError("Analysis aborted: sample size too small for Z-Test.")

    # --- Get reference population mean ---
    print(f"\nStep 2: Enter the REFERENCE (population) mean from literature.")
    print(f"        Example: WHO normal mean BP = 85, normal heart rate = 72")
    while True:
        try:
            mu_0 = float(input("Enter reference population mean (mu_0): ").strip())
            break
        except ValueError:
            print("[ERROR] Please enter a valid number.")

    # --- Get population std dev (optional) ---
    print(f"\nStep 3: Enter the POPULATION standard deviation (optional).")
    print(f"        Press ENTER to use the sample std dev (valid for n >= 30).")
    sigma_input = input("Enter population std dev (or press ENTER to skip): ").strip()

    if sigma_input == "":
        sigma = data.std(ddof=1)
        sigma_source = "Sample std dev (used because population std dev not provided)"
    else:
        try:
            sigma = float(sigma_input)
            if sigma <= 0:
                raise ValueError("Standard deviation must be positive.")
            sigma_source = f"User-provided population std dev: {sigma}"
        except ValueError as e:
            raise ValueError(f"Invalid standard deviation: {e}")

    # --- Compute Z-Statistic ---
    x_bar = data.mean()
    se = sigma / np.sqrt(n)            # Standard error of the mean
    z_stat = (x_bar - mu_0) / se      # Z-statistic

    # --- Two-tailed p-value from standard normal distribution ---
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # --- 95% Confidence Interval for sample mean ---
    z_critical = stats.norm.ppf(0.975)  # 1.96 for 95% CI
    ci_lower = x_bar - z_critical * se
    ci_upper = x_bar + z_critical * se

    # --- Effect Size (Cohen's d equivalent: d = (x_bar - mu_0) / sigma) ---
    cohens_d = (x_bar - mu_0) / sigma
    effect_label = interpret_cohens_d_label(abs(cohens_d))

    # --- Interpretation ---
    interpretation = interpret_p(p_value)
    direction = "higher than" if x_bar > mu_0 else "lower than"

    # --- Build Report ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 70,
        "VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "=" * 70,
        f"Test              : One-Sample Z-Test",
        f"Date/Time         : {timestamp}",
        f"Column Tested     : {col}",
        f"Reference Mean    : {mu_0}",
        "=" * 70,
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 70,
        f"  N                     : {n}",
        f"  Sample Mean (x̄)      : {x_bar:.4f}",
        f"  Sample Median         : {data.median():.4f}",
        f"  Sample Std Dev        : {data.std(ddof=1):.4f}",
        f"  Min                   : {data.min():.4f}",
        f"  Max                   : {data.max():.4f}",
        f"  Std Dev Used          : {sigma:.4f}",
        f"  Source of Std Dev     : {sigma_source}",
        "",
        "ASSUMPTION CHECKS",
        "-" * 70,
        f"  Sample Size (n)       : {n} {'✔ (>= 30, CLT applies)' if n >= 30 else '⚠ (< 30, consider T-Test)'}",
        "",
        "TEST RESULTS",
        "-" * 70,
        f"  Reference Mean (μ₀)  : {mu_0}",
        f"  Sample Mean (x̄)      : {x_bar:.4f}",
        f"  Standard Error (SE)   : {se:.4f}",
        f"  Z-Statistic           : {z_stat:.4f}",
        f"  P-Value (two-tailed)  : {p_value:.6f}",
        f"  95% CI for Mean       : [{ci_lower:.4f}, {ci_upper:.4f}]",
        "",
        "EFFECT SIZE",
        "-" * 70,
        f"  Cohen's d             : {cohens_d:.4f}",
        f"  Magnitude             : {effect_label}",
        "  Reference: Small=0.2 | Medium=0.5 | Large=0.8",
        "",
        "INTERPRETATION",
        "-" * 70,
        f"  {interpretation}",
        "",
        "  Medical Conclusion:",
        f"  The sample mean {col} ({x_bar:.4f}) is {direction}",
        f"  the reference population mean ({mu_0}).",
        f"  (Z = {z_stat:.4f}, p = {p_value:.6f})",
        "",
        f"  {'[✔] Significant deviation from reference mean.' if p_value < 0.05 else '[✘] No significant deviation from reference mean.'}",
        "",
        "  HYPOTHESIS:",
        f"  H₀: μ = {mu_0} (No difference from reference)",
        f"  H₁: μ ≠ {mu_0} (Significant difference from reference)",
        f"  Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to Reject H₀'}",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    print("\n" + report)
    save_report(report, "one_sample_z_test_result.txt")
    return report


# ==============================================================================
# TWO-SAMPLE Z-TEST FOR PROPORTIONS
# ==============================================================================

def run_two_sample_z_test_proportions(df: pd.DataFrame) -> str:
    """
    Execute Two-Sample Z-Test for Proportions.

    Compares the proportion of a binary outcome between two independent groups.
    This is the most clinically useful Z-test — comparing event rates
    (mortality, recovery, readmission) between two patient groups.

    Workflow:
        1. User selects binary outcome column (e.g., 'death': 0/1).
        2. User selects binary grouping column (e.g., 'sex': male/female).
        3. Compute proportion (p1, p2) in each group.
        4. Check large-sample assumption: n*p >= 5 and n*(1-p) >= 5.
        5. Compute pooled proportion.
        6. Compute Z-statistic for difference in proportions.
        7. Compute two-tailed p-value.
        8. Compute 95% CI for difference in proportions.
        9. Compute Cohen's h effect size for proportions.
        10. Display and save report.

    Returns:
        str: Full formatted report.
    """
    print("\n" + "=" * 60)
    print("TWO-SAMPLE Z-TEST FOR PROPORTIONS")
    print("=" * 60)
    print("PURPOSE: Compare the proportion (rate/%) of a binary")
    print("         outcome between two independent groups.")
    print("         Example: mortality rate in Drug vs Placebo groups.")
    print("=" * 60)

    display_columns(df)

    # --- Select binary outcome column ---
    print("\nStep 1: Select the BINARY OUTCOME column (must be 0/1 or two categories).")
    print("        SUPPORT2 examples: 'death' (0=alive, 1=dead)")
    outcome_col = input("Enter column name for binary outcome: ").strip()
    outcome_vals = validate_binary_column(df, outcome_col)

    # --- Select grouping column ---
    print("\nStep 2: Select the GROUPING column (must have exactly 2 groups).")
    print("        SUPPORT2 examples: 'sex' (male/female), 'diabetes' (yes/no)")
    group_col = input("Enter column name for grouping variable: ").strip()
    groups = validate_binary_column(df, group_col)

    # --- Handle missing values ---
    df_clean = handle_missing_values(df, [outcome_col, group_col])

    # --- Identify the "event" value (1 or the higher/positive value) ---
    # Automatically detect which value represents the event (1 or 'yes'/'true')
    event_val = _detect_event_value(outcome_vals, outcome_col)
    print(f"\n[INFO] Event value detected as: '{event_val}' (treated as the outcome/event)")
    print(f"       If incorrect, ensure your column uses 0/1 encoding.")

    # --- Split data by groups ---
    g1_data = df_clean[df_clean[group_col] == groups[0]][outcome_col]
    g2_data = df_clean[df_clean[group_col] == groups[1]][outcome_col]

    n1 = len(g1_data)
    n2 = len(g2_data)

    if n1 < 10 or n2 < 10:
        raise ValueError(
            f"Each group must have at least 10 observations for Z-Test. "
            f"Group '{groups[0]}': n={n1}, Group '{groups[1]}': n={n2}."
        )

    # --- Compute proportions ---
    count1 = (g1_data == event_val).sum()
    count2 = (g2_data == event_val).sum()
    p1 = count1 / n1
    p2 = count2 / n2

    # --- Check large-sample assumption: n*p >= 5 and n*(1-p) >= 5 ---
    assumption_g1_ok = (n1 * p1 >= 5) and (n1 * (1 - p1) >= 5)
    assumption_g2_ok = (n2 * p2 >= 5) and (n2 * (1 - p2) >= 5)

    assumption_note_g1 = (
        f"[✔] n*p={n1*p1:.1f} and n*(1-p)={n1*(1-p1):.1f} — both ≥ 5"
        if assumption_g1_ok else
        f"[⚠] n*p={n1*p1:.1f} or n*(1-p)={n1*(1-p1):.1f} — condition not met"
    )
    assumption_note_g2 = (
        f"[✔] n*p={n2*p2:.1f} and n*(1-p)={n2*(1-p2):.1f} — both ≥ 5"
        if assumption_g2_ok else
        f"[⚠] n*p={n2*p2:.1f} or n*(1-p)={n2*(1-p2):.1f} — condition not met"
    )

    if not assumption_g1_ok or not assumption_g2_ok:
        print("\n[WARNING] Large-sample assumption may be violated.")
        print("          Consider using Fisher's Exact Test for small proportions.")

    # --- Pooled proportion (under H0: p1 = p2) ---
    p_pooled = (count1 + count2) / (n1 + n2)

    # --- Standard error under H0 ---
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    if se_pooled == 0:
        raise ValueError(
            "Standard error is zero. This may occur if all outcomes in "
            "one or both groups are identical (e.g., no deaths in a group)."
        )

    # --- Z-Statistic ---
    z_stat = (p1 - p2) / se_pooled

    # --- Two-tailed p-value ---
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # --- 95% CI for difference in proportions (unpooled SE for CI) ---
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_critical = stats.norm.ppf(0.975)
    diff = p1 - p2
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    # --- Cohen's h effect size for proportions ---
    # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    effect_label = interpret_cohens_h_label(abs(cohens_h))

    # --- Relative Risk (RR) ---
    rr = p1 / p2 if p2 != 0 else float("inf")

    # --- Odds Ratio (OR) ---
    odds1 = p1 / (1 - p1) if p1 != 1 else float("inf")
    odds2 = p2 / (1 - p2) if p2 != 1 else float("inf")
    or_val = odds1 / odds2 if odds2 != 0 else float("inf")

    # --- Interpretation ---
    interpretation = interpret_p(p_value)
    higher_group = groups[0] if p1 > p2 else groups[1]
    higher_pct = max(p1, p2) * 100
    lower_pct = min(p1, p2) * 100

    # --- Build Report ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 70,
        "VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "=" * 70,
        f"Test              : Two-Sample Z-Test for Proportions",
        f"Date/Time         : {timestamp}",
        f"Outcome Column    : {outcome_col}  (event = '{event_val}')",
        f"Group Column      : {group_col}",
        "=" * 70,
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 70,
        f"  {'Group':<25} {'N':>6}  {'Events':>8}  {'Proportion':>12}  {'%':>8}",
        f"  {'-'*65}",
        f"  {str(groups[0]):<25} {n1:>6}  {count1:>8}  {p1:>12.4f}  {p1*100:>7.2f}%",
        f"  {str(groups[1]):<25} {n2:>6}  {count2:>8}  {p2:>12.4f}  {p2*100:>7.2f}%",
        f"  {'TOTAL':<25} {n1+n2:>6}  {count1+count2:>8}  {p_pooled:>12.4f}  {p_pooled*100:>7.2f}%",
        "",
        "ASSUMPTION CHECKS",
        "-" * 70,
        f"  Large-sample rule (n×p ≥ 5 and n×(1-p) ≥ 5):",
        f"    Group '{groups[0]}': {assumption_note_g1}",
        f"    Group '{groups[1]}': {assumption_note_g2}",
        "",
        "TEST RESULTS",
        "-" * 70,
        f"  Pooled Proportion     : {p_pooled:.4f}",
        f"  Difference (p1 - p2)  : {diff:.4f}  ({groups[0]} - {groups[1]})",
        f"  Z-Statistic           : {z_stat:.4f}",
        f"  P-Value (two-tailed)  : {p_value:.6f}",
        f"  95% CI for Difference : [{ci_lower:.4f}, {ci_upper:.4f}]",
        "",
        "ADDITIONAL METRICS",
        "-" * 70,
        f"  Relative Risk (RR)    : {rr:.4f}",
        f"  Odds Ratio (OR)       : {or_val:.4f}",
        "",
        "EFFECT SIZE",
        "-" * 70,
        f"  Cohen's h             : {cohens_h:.4f}",
        f"  Magnitude             : {effect_label}",
        "  Reference: Small=0.2 | Medium=0.5 | Large=0.8",
        "",
        "INTERPRETATION",
        "-" * 70,
        f"  {interpretation}",
        "",
        "  Medical Conclusion:",
        f"  The proportion of '{event_val}' (outcome) in group '{groups[0]}' ({p1*100:.2f}%)",
        f"  {'differs significantly from' if p_value < 0.05 else 'does not differ significantly from'}",
        f"  group '{groups[1]}' ({p2*100:.2f}%).",
        f"  (Z = {z_stat:.4f}, p = {p_value:.6f})",
        "",
        "  HYPOTHESIS:",
        f"  H₀: p₁ = p₂  (No difference in proportions between groups)",
        f"  H₁: p₁ ≠ p₂  (Significant difference in proportions)",
        f"  Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to Reject H₀'}",
        "",
        "  NOTE: If large-sample assumptions are violated, use Fisher's Exact Test.",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    print("\n" + report)
    save_report(report, "two_sample_z_test_proportions_result.txt")
    return report


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _detect_event_value(unique_vals: list, col_name: str):
    """
    Automatically detect which value represents the 'event' (positive outcome).

    Logic:
        - If values are 0 and 1 → return 1
        - If values are strings → return the one that looks like 'yes', 'true', 'dead', 'death'
        - Otherwise → return the second (higher) value and warn the user.

    Parameters:
        unique_vals (list): Sorted list of 2 unique values.
        col_name (str): Column name for context.

    Returns:
        The value considered as the event.
    """
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
        return 1

    # Check for common positive labels
    positive_keywords = {"yes", "true", "dead", "death", "1", "positive", "event"}
    for val in unique_vals:
        if str(val).strip().lower() in positive_keywords:
            return val

    # Default: return the higher/second value with a warning
    print(f"\n[WARNING] Could not auto-detect event value for '{col_name}'.")
    print(f"          Defaulting to '{unique_vals[-1]}' as the event.")
    print(f"          Unique values found: {unique_vals}")
    return unique_vals[-1]


def interpret_cohens_d_label(d: float) -> str:
    """Return effect size label for Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_cohens_h_label(h: float) -> str:
    """
    Return effect size label for Cohen's h (used for proportions).

    Cohen's h:
        Small  : 0.20
        Medium : 0.50
        Large  : 0.80
    """
    h = abs(h)
    if h < 0.20:
        return "Negligible"
    elif h < 0.50:
        return "Small"
    elif h < 0.80:
        return "Medium"
    else:
        return "Large"


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Main interactive entry point for the Z-Test module.
    Guides the user through CSV loading, test selection, and column selection.
    """
    print("\n" + "=" * 70)
    print("  VM-WIT-STATS | VM Medical College")
    print("  Module: Z-Test Analysis")
    print("=" * 70)

    # --- Step 1: Load CSV ---
    csv_path = input("\nEnter the full path to your CSV file: ").strip()
    try:
        df = load_csv(csv_path)
        print(f"\n[✔] Dataset loaded successfully. Shape: {df.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # --- Step 2: Choose Test Type ---
    print("\n" + "=" * 60)
    print("SELECT Z-TEST TYPE:")
    print("  1 → One-Sample Z-Test")
    print("      (Sample mean vs known reference/population mean)")
    print("  2 → Two-Sample Z-Test for Proportions")
    print("      (Compare event rates between two groups)")
    print("=" * 60)

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("[ERROR] Invalid choice. Please enter 1 or 2.")

    # --- Step 3: Run Selected Test ---
    try:
        if choice == "1":
            run_one_sample_z_test(df)
        else:
            run_two_sample_z_test_proportions(df)
    except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\n[✔] Z-Test analysis complete.")


if __name__ == "__main__":
    main()