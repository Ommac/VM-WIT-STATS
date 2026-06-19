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
    Load a CSV file into a pandas DataFrame.

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
    """Display all column names with their data types."""
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
    """Raise an error if the column is not numeric."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' is not numeric (dtype: {df[col].dtype}). "
            "Please select a continuous numeric column."
        )


def validate_group_column(df: pd.DataFrame, col: str) -> list:
    """
    Validate that the grouping column has exactly 2 unique non-null groups.

    Returns:
        list: The two unique group labels.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    groups = df[col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"Column '{col}' has {len(groups)} unique value(s): {list(groups)}. "
            "Independent T-Test requires exactly 2 groups."
        )
    return sorted(groups)


def handle_missing_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Drop rows with missing values in the specified columns and report count.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        cols (list): Columns to check for missing values.

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


def cohens_d_independent(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size for independent groups.

    Cohen's d interpretation:
        Small  : 0.2
        Medium : 0.5
        Large  : 0.8

    Parameters:
        group1, group2 (pd.Series): Data from each group.

    Returns:
        float: Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std(ddof=1) ** 2 + (n2 - 1) * group2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cohens_d_paired(differences: pd.Series) -> float:
    """
    Calculate Cohen's d effect size for paired data.

    Parameters:
        differences (pd.Series): Series of paired differences (post - pre).

    Returns:
        float: Cohen's d value.
    """
    if differences.std(ddof=1) == 0:
        return 0.0
    return differences.mean() / differences.std(ddof=1)


def interpret_cohens_d(d: float) -> str:
    """Return a plain-language label for Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def normality_check(data: pd.Series, label: str) -> dict:
    """
    Run Shapiro-Wilk normality test on a data series.

    Note: Shapiro-Wilk is reliable for n <= 5000.
          For larger samples, normality is less critical due to the Central Limit Theorem.

    Parameters:
        data (pd.Series): Data to test.
        label (str): Label for display purposes.

    Returns:
        dict: {'stat', 'p', 'normal', 'note'}
    """
    n = len(data)
    if n < 3:
        return {"stat": None, "p": None, "normal": None,
                "note": f"[{label}] Too few observations (n={n}) for Shapiro-Wilk."}
    if n > 5000:
        return {"stat": None, "p": None, "normal": True,
                "note": f"[{label}] n={n} > 5000. Normality assumed via CLT; Shapiro-Wilk skipped."}
    stat, p = stats.shapiro(data)
    normal = p > 0.05
    interpretation = "Normally distributed" if normal else "NOT normally distributed"
    return {
        "stat": stat,
        "p": p,
        "normal": normal,
        "note": f"[{label}] Shapiro-Wilk: W={stat:.4f}, p={p:.4f} → {interpretation}"
    }


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
    """Save the analysis report to the outputs/ directory."""
    create_output_dir()
    filepath = os.path.join("outputs", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[✔] Report saved to: {filepath}")

# INDEPENDENT T-TEST

def run_independent_t_test(df: pd.DataFrame) -> str:
    """
    Execute Independent (Two-Sample) T-Test.

    Workflow:
        1. User selects continuous outcome column and binary group column.
        2. Validate inputs and handle missing values.
        3. Check normality in each group (Shapiro-Wilk).
        4. Check homogeneity of variance (Levene's test).
        5. Run Independent T-Test (equal or unequal variance version).
        6. Compute Cohen's d effect size.
        7. Compute 95% confidence interval for mean difference.
        8. Display and return full report.

    Returns:
        str: Complete formatted report.
    """
    print("\n" + "=" * 60)
    print("INDEPENDENT T-TEST (Two-Sample T-Test)")
    print("=" * 60)
    print("PURPOSE: Compare the mean of a continuous variable between")
    print("         two independent groups.")
    print("=" * 60)

    display_columns(df)

    # --- Select continuous outcome column ---
    print("\nStep 1: Select the CONTINUOUS OUTCOME (dependent) column.")
    print("        Example: 'blood_pressure', 'hba1c', 'pain_score'")
    outcome_col = input("Enter column name for outcome variable: ").strip()
    validate_numeric_column(df, outcome_col)

    # --- Select grouping column ---
    print("\nStep 2: Select the GROUPING column (must have exactly 2 groups).")
    print("        Example: 'gender' (Male/Female), 'treatment' (Drug/Placebo)")
    group_col = input("Enter column name for grouping variable: ").strip()
    groups = validate_group_column(df, group_col)

    # --- Clean missing values ---
    df_clean = handle_missing_values(df, [outcome_col, group_col])

    # --- Split into two groups ---
    group1_data = df_clean[df_clean[group_col] == groups[0]][outcome_col]
    group2_data = df_clean[df_clean[group_col] == groups[1]][outcome_col]

    print(f"\n[INFO] Group 1 → '{groups[0]}': n = {len(group1_data)}")
    print(f"[INFO] Group 2 → '{groups[1]}': n = {len(group2_data)}")

    if len(group1_data) < 2 or len(group2_data) < 2:
        raise ValueError("Each group must have at least 2 observations.")

    # --- Descriptive Statistics ---
    desc = {
        groups[0]: group1_data.describe(),
        groups[1]: group2_data.describe()
    }

    # --- Normality Check ---
    norm1 = normality_check(group1_data, str(groups[0]))
    norm2 = normality_check(group2_data, str(groups[1]))

    # --- Levene's Test for Homogeneity of Variance ---
    lev_stat, lev_p = stats.levene(group1_data, group2_data)
    equal_var = lev_p > 0.05
    variance_note = (
        "Variances are EQUAL (Levene's p ≥ 0.05) → Using standard T-Test."
        if equal_var else
        "Variances are UNEQUAL (Levene's p < 0.05) → Using Welch's T-Test."
    )

    # --- Independent T-Test ---
    # equal_var=True → Student's T-Test; equal_var=False → Welch's T-Test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
    df_freedom = (
        len(group1_data) + len(group2_data) - 2
        if equal_var
        else None  # Welch's df is Satterthwaite approximation
    )

    # --- 95% Confidence Interval for Mean Difference ---
    mean_diff = group1_data.mean() - group2_data.mean()
    se_diff = np.sqrt(
        group1_data.var(ddof=1) / len(group1_data) +
        group2_data.var(ddof=1) / len(group2_data)
    )
    alpha = 0.05
    # Use t-distribution for CI
    if equal_var and df_freedom:
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_freedom)
    else:
        # Welch-Satterthwaite df approximation
        n1, n2 = len(group1_data), len(group2_data)
        s1, s2 = group1_data.var(ddof=1), group2_data.var(ddof=1)
        welch_df = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df=welch_df)

    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # --- Cohen's d Effect Size ---
    d = cohens_d_independent(group1_data, group2_data)
    effect_label = interpret_cohens_d(d)

    # --- Interpretation ---
    interpretation = interpret_p(p_value)
    test_type = "Student's Independent T-Test" if equal_var else "Welch's Independent T-Test"

    # ==================================================================
    # BUILD REPORT
    # ==================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 70,
        "VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "=" * 70,
        f"Test           : {test_type}",
        f"Date/Time      : {timestamp}",
        f"Outcome Column : {outcome_col}",
        f"Group Column   : {group_col}",
        "=" * 70,
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 70,
    ]

    for grp in groups:
        grp_data = df_clean[df_clean[group_col] == grp][outcome_col]
        report_lines += [
            f"  Group: {grp}",
            f"    N        : {len(grp_data)}",
            f"    Mean     : {grp_data.mean():.4f}",
            f"    Median   : {grp_data.median():.4f}",
            f"    Std Dev  : {grp_data.std(ddof=1):.4f}",
            f"    Min      : {grp_data.min():.4f}",
            f"    Max      : {grp_data.max():.4f}",
            "",
        ]

    report_lines += [
        "ASSUMPTION CHECKS",
        "-" * 70,
        "  Normality (Shapiro-Wilk):",
        f"    {norm1['note']}",
        f"    {norm2['note']}",
        "",
        "  Homogeneity of Variance (Levene's Test):",
        f"    Levene Statistic : {lev_stat:.4f}",
        f"    Levene p-value   : {lev_p:.4f}",
        f"    {variance_note}",
        "",
        "TEST RESULTS",
        "-" * 70,
        f"  Test Applied      : {test_type}",
        f"  T-Statistic       : {t_stat:.4f}",
        f"  Degrees of Freedom: {df_freedom if df_freedom else 'Welch approximation'}",
        f"  P-Value           : {p_value:.6f}",
        f"  Mean Difference   : {mean_diff:.4f}  ({groups[0]} - {groups[1]})",
        f"  95% CI for Diff   : [{ci_lower:.4f}, {ci_upper:.4f}]",
        "",
        "EFFECT SIZE",
        "-" * 70,
        f"  Cohen's d         : {d:.4f}",
        f"  Magnitude         : {effect_label}",
        "  Reference: Small=0.2 | Medium=0.5 | Large=0.8",
        "",
        "INTERPRETATION",
        "-" * 70,
        f"  {interpretation}",
        "",
        "  Medical Conclusion:",
        f"  The mean {outcome_col} for group '{groups[0]}' "
        f"({group1_data.mean():.4f}) "
        f"{'differs significantly from' if p_value < 0.05 else 'does not differ significantly from'}",
        f"  group '{groups[1]}' ({group2_data.mean():.4f}).",
        f"  (t = {t_stat:.4f}, p = {p_value:.6f})",
        "",
        "  NOTE: If normality assumptions were violated, consider using",
        "        the Mann-Whitney U Test (non-parametric alternative).",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    report = "\n".join(report_lines)

    # Print to console
    print("\n" + report)

    # Save to file
    save_report(report, "independent_t_test_result.txt")

    return report


# ==============================================================================
# PAIRED T-TEST
# ==============================================================================

def run_paired_t_test(df: pd.DataFrame) -> str:
    """
    Execute Paired (Dependent) T-Test.

    Workflow:
        1. User selects two paired numeric columns (e.g., pre and post measurements).
        2. Validate inputs and handle missing values.
        3. Compute differences (col2 - col1).
        4. Check normality of differences (Shapiro-Wilk).
        5. Run Paired T-Test.
        6. Compute Cohen's d and 95% CI for mean difference.
        7. Display and return full report.

    Returns:
        str: Complete formatted report.
    """
    print("\n" + "=" * 60)
    print("PAIRED T-TEST (Dependent / Repeated Measures T-Test)")
    print("=" * 60)
    print("PURPOSE: Compare means of TWO RELATED measurements")
    print("         on the SAME subjects (e.g., pre vs post treatment).")
    print("=" * 60)

    display_columns(df)

    # --- Select Pre/Baseline column ---
    print("\nStep 1: Select the FIRST measurement column (e.g., 'BP_before', 'pre_treatment').")
    col1 = input("Enter column name for first (baseline/pre) measurement: ").strip()
    validate_numeric_column(df, col1)

    # --- Select Post column ---
    print("\nStep 2: Select the SECOND measurement column (e.g., 'BP_after', 'post_treatment').")
    col2 = input("Enter column name for second (follow-up/post) measurement: ").strip()
    validate_numeric_column(df, col2)

    if col1 == col2:
        raise ValueError("The two columns must be different.")

    # --- Clean missing values ---
    df_clean = handle_missing_values(df, [col1, col2])

    pre = df_clean[col1]
    post = df_clean[col2]
    n = len(pre)

    print(f"\n[INFO] Number of paired observations: {n}")

    if n < 3:
        raise ValueError("Paired T-Test requires at least 3 paired observations.")

    # --- Compute Differences ---
    differences = post - pre  # Convention: post minus pre

    # --- Normality of Differences ---
    norm_diff = normality_check(differences, "Differences (post - pre)")

    # --- Paired T-Test ---
    t_stat, p_value = stats.ttest_rel(pre, post)
    df_freedom = n - 1

    # --- 95% CI for Mean Difference ---
    mean_diff = differences.mean()
    se_diff = differences.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=df_freedom)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # --- Cohen's d ---
    d = cohens_d_paired(differences)
    effect_label = interpret_cohens_d(d)

    # --- Interpretation ---
    interpretation = interpret_p(p_value)

    # ==================================================================
    # BUILD REPORT
    # ==================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 70,
        "VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit",
        "=" * 70,
        f"Test              : Paired T-Test",
        f"Date/Time         : {timestamp}",
        f"Measurement 1     : {col1} (Baseline / Pre)",
        f"Measurement 2     : {col2} (Follow-up / Post)",
        "=" * 70,
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 70,
        f"  N (paired pairs)      : {n}",
        f"  {col1} Mean           : {pre.mean():.4f}",
        f"  {col1} Std Dev        : {pre.std(ddof=1):.4f}",
        f"  {col2} Mean           : {post.mean():.4f}",
        f"  {col2} Std Dev        : {post.std(ddof=1):.4f}",
        "",
        "  Differences (post - pre):",
        f"    Mean Difference     : {mean_diff:.4f}",
        f"    Std Dev of Diff     : {differences.std(ddof=1):.4f}",
        f"    Min Difference      : {differences.min():.4f}",
        f"    Max Difference      : {differences.max():.4f}",
        "",
        "ASSUMPTION CHECK",
        "-" * 70,
        "  Normality of Differences (Shapiro-Wilk):",
        f"    {norm_diff['note']}",
        "",
        "TEST RESULTS",
        "-" * 70,
        f"  T-Statistic           : {t_stat:.4f}",
        f"  Degrees of Freedom    : {df_freedom}",
        f"  P-Value               : {p_value:.6f}",
        f"  Mean Difference       : {mean_diff:.4f}  ({col2} - {col1})",
        f"  95% CI for Difference : [{ci_lower:.4f}, {ci_upper:.4f}]",
        "",
        "EFFECT SIZE",
        "-" * 70,
        f"  Cohen's d             : {d:.4f}",
        f"  Magnitude             : {effect_label}",
        "  Reference: Small=0.2 | Medium=0.5 | Large=0.8",
        "",
        "INTERPRETATION",
        "-" * 70,
        f"  {interpretation}",
        "",
        "  Medical Conclusion:",
        f"  The mean {col2} ({post.mean():.4f}) "
        f"{'is significantly different from' if p_value < 0.05 else 'is not significantly different from'}",
        f"  the mean {col1} ({pre.mean():.4f}).",
        f"  (t = {t_stat:.4f}, df = {df_freedom}, p = {p_value:.6f})",
        "",
        f"  {'[✔] Treatment effect detected.' if p_value < 0.05 else '[✘] No significant treatment effect detected.'}",
        "",
        "  NOTE: If differences are NOT normally distributed, consider using",
        "        the Wilcoxon Signed-Rank Test (non-parametric alternative).",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    report = "\n".join(report_lines)

    # Print to console
    print("\n" + report)

    # Save to file
    save_report(report, "paired_t_test_result.txt")

    return report


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Main interactive entry point for the T-Test module.
    Guides the user through selecting CSV, test type, and columns.
    """
    print("\n" + "=" * 70)
    print("  VM-WIT-STATS | VM Medical College")
    print("  Module: T-Test Analysis")
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
    print("SELECT T-TEST TYPE:")
    print("  1 → Independent T-Test  (Two separate groups)")
    print("  2 → Paired T-Test       (Same subjects, two time points)")
    print("=" * 60)

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("[ERROR] Invalid choice. Please enter 1 or 2.")

    # --- Step 3: Run Selected Test ---
    try:
        if choice == "1":
            run_independent_t_test(df)
        else:
            run_paired_t_test(df)
    except (ValueError, TypeError, KeyError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\n[✔] T-Test analysis complete.")


if __name__ == "__main__":
    main()