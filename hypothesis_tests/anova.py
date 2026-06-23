import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import tukey_hsd
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
        filename : Target filename (e.g. 'anova_test_result.txt').
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
        ValueError       : Non-CSV extension or empty file.
        FileNotFoundError: Path does not exist.
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


def validate_numeric_column(df: pd.DataFrame, col: str) -> None:
    """
    Validate that a column exists and contains numeric data.

    Raises:
        ValueError : Column not in dataframe.
        TypeError  : Column is not numeric.
    """
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' has dtype '{df[col].dtype}'. "
            "ANOVA requires a numeric (continuous) outcome column."
        )


def validate_group_column(df: pd.DataFrame, col: str) -> list:
    """
    Validate that a group column exists and contains at least 3 unique groups.

    Parameters:
        df  : Dataset.
        col : Column name for grouping.

    Returns:
        Sorted list of unique group labels.

    Raises:
        ValueError : Column missing or fewer than 3 groups.
    """
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    unique_groups = sorted(df[col].dropna().unique(), key=str)
    if len(unique_groups) < 3:
        raise ValueError(
            f"Column '{col}' has only {len(unique_groups)} unique group(s): "
            f"{unique_groups}. "
            "One-Way ANOVA requires at least 3 groups. "
            "For 2 groups, use the Independent T-Test or Mann-Whitney U Test."
        )
    return unique_groups


def clean_data(df: pd.DataFrame,
               outcome_col: str,
               group_col: str) -> pd.DataFrame:
    """
    Produce a clean working copy of the dataset.

    Steps:
        1. Keep only the two relevant columns.
        2. Drop rows where either column is null/NaN.
        3. Replace infinite values in outcome with NaN then drop.
        4. Cast outcome to float64.
        5. Strip whitespace from group labels.

    Parameters:
        df          : Raw dataset.
        outcome_col : Continuous outcome column name.
        group_col   : Categorical group column name.

    Returns:
        Cleaned DataFrame with only [outcome_col, group_col].
    """
    original_n = len(df)
    working    = df[[outcome_col, group_col]].copy()

    # Replace ±inf in outcome with NaN (CoW-safe: assign to new series first)
    outcome_series = pd.to_numeric(working[outcome_col], errors="coerce")
    outcome_series = outcome_series.replace([np.inf, -np.inf], np.nan)
    working = working.copy()
    working[outcome_col] = outcome_series

    # Strip whitespace from string group labels (handles object and StringDtype)
    if pd.api.types.is_string_dtype(working[group_col]) or working[group_col].dtype == object:
        working[group_col] = working[group_col].str.strip()

    # Drop rows with any NaN in either column (CoW-safe: reassign)
    working = working.dropna(subset=[outcome_col, group_col])

    dropped = original_n - len(working)
    if dropped > 0:
        print(f"\n  [INFO] {dropped:,} row(s) removed (missing / infinite values).")
    else:
        print(f"\n  [INFO] No missing or infinite values found.")
    print(f"  [INFO] Rows used for analysis: {len(working):,}")
    return working


# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

def compute_group_descriptives(data: pd.Series,
                                label: str) -> dict:
    """
    Compute a full set of descriptive statistics for a single group.

    Parameters:
        data  : Series of numeric outcome values for this group.
        label : Group label (for display).

    Returns:
        Dictionary of statistics.
    """
    n        = len(data)
    mean_val = data.mean()
    median   = data.median()
    std      = data.std(ddof=1) if n > 1 else np.nan
    se       = std / np.sqrt(n) if n > 1 else np.nan
    q1       = data.quantile(0.25)
    q3       = data.quantile(0.75)
    iqr      = q3 - q1
    skew     = float(data.skew())   if n > 2 else np.nan
    kurt     = float(data.kurtosis()) if n > 3 else np.nan

    return {
        "label"  : label,
        "n"      : n,
        "mean"   : mean_val,
        "median" : median,
        "std"    : std,
        "se"     : se,
        "min"    : data.min(),
        "max"    : data.max(),
        "q1"     : q1,
        "q3"     : q3,
        "iqr"    : iqr,
        "skew"   : skew,
        "kurt"   : kurt,
    }


def format_descriptives_block(desc_list: list,
                               outcome_col: str) -> list:
    """
    Format all group descriptives into a list of report lines.

    Parameters:
        desc_list   : List of dicts from compute_group_descriptives().
        outcome_col : Name of the outcome column (for header).

    Returns:
        List of formatted strings.
    """
    lines = [
        "",
        "DESCRIPTIVE STATISTICS",
        THIN_DIVIDER,
        f"  Outcome Variable : {outcome_col}",
        "",
    ]

    def _fmt(val, decimals=4):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        return f"{val:.{decimals}f}"

    for d in desc_list:
        lines += [
            f"  Group            : {d['label']}",
            f"  {'N':<20} : {d['n']:,}",
            f"  {'Mean':<20} : {_fmt(d['mean'])}",
            f"  {'Median':<20} : {_fmt(d['median'])}",
            f"  {'Std Deviation':<20} : {_fmt(d['std'])}",
            f"  {'Std Error':<20} : {_fmt(d['se'])}",
            f"  {'Min':<20} : {_fmt(d['min'])}",
            f"  {'Max':<20} : {_fmt(d['max'])}",
            f"  {'Q1 (25th pct)':<20} : {_fmt(d['q1'])}",
            f"  {'Q3 (75th pct)':<20} : {_fmt(d['q3'])}",
            f"  {'IQR':<20} : {_fmt(d['iqr'])}",
            f"  {'Skewness':<20} : {_fmt(d['skew'])}",
            f"  {'Kurtosis':<20} : {_fmt(d['kurt'])}",
            "",
        ]
    return lines


# ==============================================================================
# ASSUMPTION CHECKS
# ==============================================================================

def check_normality(data: pd.Series, label: str) -> dict:
    """
    Run Shapiro-Wilk normality test on a single group.

    Rules:
        n < 3    → cannot run (insufficient data).
        n > 5000 → skip (CLT; normality not assessed).
        else     → run scipy.stats.shapiro().

    Parameters:
        data  : Group values.
        label : Group label.

    Returns:
        Dict with keys: label, n, stat, p, normal, note.
    """
    n = len(data)
    base = {"label": label, "n": n}

    if n < 3:
        return {**base, "stat": None, "p": None, "normal": None,
                "note": f"Insufficient data (n={n} < 3). Normality not assessed."}

    if n > 5000:
        return {**base, "stat": None, "p": None, "normal": True,
                "note": f"n={n:,} > 5000. Normality not assessed (CLT applies)."}

    w_stat, p_val = stats.shapiro(data)
    normal = p_val > 0.05
    verdict = "Normally distributed (p ≥ 0.05)" if normal else "NOT normally distributed (p < 0.05)"
    return {
        **base,
        "stat"  : w_stat,
        "p"     : p_val,
        "normal": normal,
        "note"  : f"Shapiro-Wilk: W = {w_stat:.4f}, p = {p_val:.4f} → {verdict}",
    }


def check_levene(groups_data: list, group_labels: list) -> dict:
    """
    Run Levene's Test for homogeneity of variance.

    Parameters:
        groups_data  : List of pd.Series, one per group.
        group_labels : Corresponding group label strings.

    Returns:
        Dict with keys: stat, p, equal_var, note.
    """
    lev_stat, lev_p = stats.levene(*groups_data)
    equal_var = lev_p >= 0.05
    verdict = (
        "Variances are EQUAL across groups (p ≥ 0.05). "
        "Homogeneity of variance assumption satisfied."
        if equal_var else
        "Variances are UNEQUAL across groups (p < 0.05). "
        "Homogeneity assumption VIOLATED — interpret results with caution."
    )
    return {
        "stat"     : lev_stat,
        "p"        : lev_p,
        "equal_var": equal_var,
        "note"     : verdict,
    }


def format_assumption_block(norm_results: list,
                             levene_result: dict) -> list:
    """
    Format assumption check results into report lines.

    Parameters:
        norm_results  : List of dicts from check_normality().
        levene_result : Dict from check_levene().

    Returns:
        List of formatted strings.
    """
    lines = [
        "",
        "ASSUMPTION CHECKS",
        THIN_DIVIDER,
        "  1. Independence of Observations",
        "     Assumed — each row represents a unique patient/subject.",
        "     Verify this holds for your study design.",
        "",
        "  2. Normality (Shapiro-Wilk Test per Group)",
    ]
    any_violated = False
    for nr in norm_results:
        flag = ""
        if nr["normal"] is False:
            flag = " ⚠"
            any_violated = True
        lines.append(f"     Group '{nr['label']}' (n={nr['n']:,}): {nr['note']}{flag}")

    if any_violated:
        lines += [
            "",
            "     ⚠ WARNING: Normality violated in one or more groups.",
            "     Consider: Kruskal-Wallis Test (non-parametric alternative).",
        ]
    else:
        lines.append("")

    lines += [
        "  3. Homogeneity of Variance (Levene's Test)",
        f"     Levene Statistic : {levene_result['stat']:.4f}",
        f"     Levene p-value   : {levene_result['p']:.4f}",
        f"     {levene_result['note']}",
    ]

    if not levene_result["equal_var"]:
        lines += [
            "",
            "     ⚠ NOTE: Welch's ANOVA or a non-parametric test may be more",
            "       appropriate when variance homogeneity is violated.",
        ]

    lines += [
        "",
        "  4. ANOVA Appropriateness",
        "     ✔ Outcome variable is continuous.",
        f"    ✔ Group variable contains ≥ 3 independent groups.",
        "     ✔ Standard One-Way ANOVA applied.",
        "",
    ]
    return lines


# ==============================================================================
# ETA SQUARED EFFECT SIZE
# ==============================================================================

def compute_eta_squared(groups_data: list) -> dict:
    """
    Compute Eta Squared (η²) effect size for One-Way ANOVA.

    Formula:
        SS_between = Σ nᵢ × (x̄ᵢ − x̄_grand)²
        SS_total   = Σ Σ (xᵢⱼ − x̄_grand)²
        η²         = SS_between / SS_total

    Interpretation:
        η² < 0.01  → Negligible
        η² < 0.06  → Small
        η² < 0.14  → Medium
        η² ≥ 0.14  → Large

    Parameters:
        groups_data : List of pd.Series, one per group.

    Returns:
        Dict with keys: ss_between, ss_within, ss_total, eta_sq, label.
    """
    all_values  = np.concatenate([g.values for g in groups_data])
    grand_mean  = all_values.mean()

    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in groups_data
    )
    ss_within = sum(
        ((g - g.mean()) ** 2).sum()
        for g in groups_data
    )
    ss_total = ss_between + ss_within

    if ss_total == 0:
        eta_sq = 0.0
    else:
        eta_sq = ss_between / ss_total

    if eta_sq < 0.01:
        label = "Negligible"
    elif eta_sq < 0.06:
        label = "Small"
    elif eta_sq < 0.14:
        label = "Medium"
    else:
        label = "Large"

    return {
        "ss_between": ss_between,
        "ss_within" : ss_within,
        "ss_total"  : ss_total,
        "eta_sq"    : eta_sq,
        "label"     : label,
    }


# ==============================================================================
# POST-HOC — TUKEY HSD
# ==============================================================================

def run_tukey_hsd(groups_data: list,
                  group_labels: list) -> list:
    """
    Run Tukey HSD pairwise post-hoc comparisons using scipy.stats.tukey_hsd.

    Only called when ANOVA p < 0.05.

    Parameters:
        groups_data  : List of pd.Series, one per group.
        group_labels : Corresponding group label strings.

    Returns:
        List of dicts, one per pair:
            group_a, group_b, mean_a, mean_b,
            mean_diff, statistic, p_adj, significant, ci_low, ci_high
    """
    result = tukey_hsd(*[g.values for g in groups_data])
    ci     = result.confidence_interval()

    pairs = []
    k = len(group_labels)
    for i in range(k):
        for j in range(i + 1, k):
            mean_a    = groups_data[i].mean()
            mean_b    = groups_data[j].mean()
            mean_diff = mean_a - mean_b
            stat      = result.statistic[i, j]
            p_adj     = result.pvalue[i, j]
            ci_low    = ci.low[i, j]
            ci_high   = ci.high[i, j]
            sig       = "Yes" if p_adj < 0.05 else "No"

            pairs.append({
                "group_a"  : group_labels[i],
                "group_b"  : group_labels[j],
                "mean_a"   : mean_a,
                "mean_b"   : mean_b,
                "mean_diff": mean_diff,
                "statistic": stat,
                "p_adj"    : p_adj,
                "sig"      : sig,
                "ci_low"   : ci_low,
                "ci_high"  : ci_high,
            })
    return pairs


def format_tukey_block(pairs: list,
                        outcome_col: str) -> list:
    """
    Format Tukey HSD pairs into report lines.

    Parameters:
        pairs       : List of pair dicts from run_tukey_hsd().
        outcome_col : Outcome column name (for context).

    Returns:
        List of formatted strings.
    """
    lines = [
        "",
        "POST-HOC ANALYSIS — TUKEY HSD",
        THIN_DIVIDER,
        "  (Performed because ANOVA was statistically significant)",
        f"  Outcome Variable : {outcome_col}",
        "",
        f"  {'Group A':<22}  {'Group B':<22}  {'Mean A':>9}  "
        f"{'Mean B':>9}  {'Diff':>9}  {'p-adj':>8}  {'Sig':>4}  {'95% CI'}",
        f"  {'-'*22}  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*4}  {'-'*22}",
    ]

    for p in pairs:
        ci_str = f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]"
        lines.append(
            f"  {str(p['group_a']):<22}  {str(p['group_b']):<22}  "
            f"{p['mean_a']:>9.4f}  {p['mean_b']:>9.4f}  "
            f"{p['mean_diff']:>+9.4f}  {p['p_adj']:>8.4f}  "
            f"{'✔' if p['sig']=='Yes' else '✘':>4}  {ci_str}"
        )

    sig_pairs = [p for p in pairs if p["sig"] == "Yes"]
    lines += [
        "",
        f"  Significant pairs (p < 0.05) : {len(sig_pairs)} of {len(pairs)}",
        "",
    ]

    if sig_pairs:
        lines.append("  Significant differences found between:")
        for p in sig_pairs:
            lines.append(
                f"    • '{p['group_a']}' vs '{p['group_b']}' "
                f"(Mean diff = {p['mean_diff']:+.4f}, p = {p['p_adj']:.4f})"
            )
    else:
        lines.append(
            "  No individual pairs reached significance at p < 0.05 "
            "despite overall ANOVA significance.\n"
            "  This can occur with many groups or borderline p-values."
        )

    lines.append("")
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


def format_interpretation_block(f_stat: float,
                                  df_between: int,
                                  df_within: int,
                                  p_value: float,
                                  eta: dict,
                                  group_labels: list,
                                  outcome_col: str,
                                  group_col: str,
                                  levene_equal: bool) -> list:
    """
    Build the interpretation and clinical conclusion section.

    Parameters:
        f_stat      : ANOVA F-statistic.
        df_between  : Degrees of freedom between groups.
        df_within   : Degrees of freedom within groups.
        p_value     : ANOVA p-value.
        eta         : Dict from compute_eta_squared().
        group_labels: List of group names.
        outcome_col : Outcome variable name.
        group_col   : Group variable name.
        levene_equal: Whether variance homogeneity holds.

    Returns:
        List of formatted strings.
    """
    significant = p_value < 0.05
    p_interp    = interpret_p(p_value)

    if significant:
        main_conclusion = (
            f"At least one group mean of '{outcome_col}' differs "
            f"significantly from the others across '{group_col}' groups."
        )
        action = (
            "Tukey HSD post-hoc analysis has been performed above to "
            "identify which specific groups differ."
        )
    else:
        main_conclusion = (
            f"No statistically significant difference was found in "
            f"'{outcome_col}' across '{group_col}' groups."
        )
        action = (
            "Post-hoc analysis was not performed (ANOVA was not significant)."
        )

    lines = [
        "",
        "MEDICAL & STATISTICAL INTERPRETATION",
        THIN_DIVIDER,
        "",
        f"  Overall Result      : {p_interp}",
        f"  ANOVA Decision      : {'Reject H₀' if significant else 'Fail to Reject H₀'}",
        "",
        "  HYPOTHESIS:",
        f"  H₀ : All group means of '{outcome_col}' are equal.",
        f"  H₁ : At least one group mean differs.",
        "",
        f"  ANOVA: F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_value:.6f}",
        "",
        "  MAIN CONCLUSION:",
        f"  {main_conclusion}",
        "",
        f"  ACTION: {action}",
        "",
        "  EFFECT SIZE (Eta Squared — η²):",
        f"    η²              = {eta['eta_sq']:.4f}",
        f"    Magnitude       = {eta['label']}",
        f"    SS Between      = {eta['ss_between']:.4f}",
        f"    SS Within       = {eta['ss_within']:.4f}",
        f"    SS Total        = {eta['ss_total']:.4f}",
        "    Reference: Negligible<0.01 | Small<0.06 | Medium<0.14 | Large≥0.14",
        "",
        "  CLINICAL RELEVANCE:",
        f"    The effect size (η² = {eta['eta_sq']:.4f}) indicates a "
        f"{eta['label'].lower()} practical effect.",
        f"    Even {'a statistically significant' if significant else 'a non-significant'} "
        f"result should be evaluated in the context",
        f"    of clinical meaningfulness, sample size, and study design.",
        "",
        "  LIMITATIONS:",
        "    • One-Way ANOVA assumes independence, normality, and homogeneity.",
        "    • Violations of these assumptions reduce reliability of results.",
        f"    {'• Levene test detected unequal variances — proceed with caution.' if not levene_equal else '• Levene test confirmed equal variances across groups.'}",
        "    • ANOVA identifies THAT a difference exists, not WHERE (use post-hoc).",
        "    • Correlation does not imply causation.",
        "",
        "  NON-PARAMETRIC ALTERNATIVE:",
        "    If normality assumptions are seriously violated, use the",
        "    Kruskal-Wallis Test (hypothesis_tests/kruskal_wallis_test.py).",
        "",
    ]
    return lines


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_anova(df: pd.DataFrame) -> str:
    """
    Execute the full One-Way ANOVA pipeline.

    Steps:
        1. Display columns.
        2. Collect and validate user inputs.
        3. Clean data.
        4. Display group summary.
        5. Compute descriptive statistics.
        6. Check assumptions.
        7. Run scipy.stats.f_oneway().
        8. Compute Eta Squared.
        9. Run Tukey HSD (if significant).
        10. Assemble and save report.

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
    print("  ONE-WAY ANOVA — INPUT SELECTION")
    print(THIN_DIVIDER)
    print("  SUPPORT2 examples:")
    print("    Outcome  : charges, meanbp, age, hrt, crea, bili")
    print("    Grouping : dzclass, race, income, sex")
    print()

    print("  Step 1: Select the CONTINUOUS OUTCOME variable.")
    outcome_col = input("  Enter outcome column name: ").strip()
    validate_numeric_column(df, outcome_col)

    print("\n  Step 2: Select the GROUP variable (must have ≥ 3 groups).")
    group_col = input("  Enter group column name: ").strip()
    group_labels = validate_group_column(df, group_col)

    print(f"\n  [INFO] Groups detected ({len(group_labels)}): {group_labels}")

    # ------------------------------------------------------------------ #
    # STEP 3 — Clean data
    # ------------------------------------------------------------------ #
    print(f"\n{THIN_DIVIDER}")
    print("  DATA CLEANING")
    print(THIN_DIVIDER)
    df_clean = clean_data(df, outcome_col, group_col)

    # ------------------------------------------------------------------ #
    # STEP 4 — Re-derive groups from cleaned data and validate group sizes
    # ------------------------------------------------------------------ #
    group_labels = sorted(df_clean[group_col].unique(), key=str)
    groups_data  = []
    removed_groups = []

    for label in group_labels:
        grp = df_clean[df_clean[group_col] == label][outcome_col].reset_index(drop=True)
        if len(grp) < 2:
            removed_groups.append((label, len(grp)))
        else:
            groups_data.append((label, grp))

    if removed_groups:
        print(f"\n  [WARNING] The following groups were excluded (n < 2):")
        for lbl, cnt in removed_groups:
            print(f"    • '{lbl}' (n = {cnt})")

    group_labels = [item[0] for item in groups_data]
    groups_series = [item[1] for item in groups_data]

    if len(groups_series) < 3:
        raise ValueError(
            f"After cleaning, only {len(groups_series)} valid group(s) remain "
            "(need at least 3 with n ≥ 2). "
            "Please check your data or choose a different group column."
        )

    # Check for constant groups (zero variance)
    constant_groups = [
        group_labels[i]
        for i, g in enumerate(groups_series)
        if g.std(ddof=1) == 0 or g.nunique() == 1
    ]
    if constant_groups:
        print(f"\n  [WARNING] Constant groups detected (zero variance): {constant_groups}")
        print("    ANOVA may produce NaN results. Results should be interpreted with caution.")

    print(f"\n{THIN_DIVIDER}")
    print("  GROUP SUMMARY")
    print(THIN_DIVIDER)
    print(f"  {'Group':<30}  {'N':>7}  {'Mean':>10}  {'Std Dev':>10}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*10}")
    for lbl, grp in zip(group_labels, groups_series):
        print(f"  {str(lbl):<30}  {len(grp):>7,}  {grp.mean():>10.4f}  {grp.std(ddof=1):>10.4f}")

    # ------------------------------------------------------------------ #
    # STEP 5 — Descriptive statistics
    # ------------------------------------------------------------------ #
    desc_list = [
        compute_group_descriptives(grp, lbl)
        for lbl, grp in zip(group_labels, groups_series)
    ]

    # ------------------------------------------------------------------ #
    # STEP 6 — Assumption checks
    # ------------------------------------------------------------------ #
    print(f"\n  [INFO] Running assumption checks ...")
    norm_results  = [check_normality(grp, lbl) for lbl, grp in zip(group_labels, groups_series)]
    levene_result = check_levene(groups_series, group_labels)

    # ------------------------------------------------------------------ #
    # STEP 7 — Run One-Way ANOVA
    # ------------------------------------------------------------------ #
    print("  [INFO] Running One-Way ANOVA ...")
    f_stat, p_value = stats.f_oneway(*groups_series)

    # Guard against NaN output (can occur with constant groups)
    if np.isnan(f_stat) or np.isnan(p_value):
        raise ValueError(
            "ANOVA returned NaN. This typically occurs when one or more groups "
            "have zero variance (all values identical). "
            "Please check your data."
        )

    k          = len(groups_series)
    N          = sum(len(g) for g in groups_series)
    df_between = k - 1
    df_within  = N - k

    # ------------------------------------------------------------------ #
    # STEP 8 — Effect size (Eta Squared)
    # ------------------------------------------------------------------ #
    eta = compute_eta_squared(groups_series)

    # ------------------------------------------------------------------ #
    # STEP 9 — Post-hoc (Tukey HSD) if significant
    # ------------------------------------------------------------------ #
    tukey_pairs = None
    if p_value < 0.05:
        print("  [INFO] ANOVA is significant. Running Tukey HSD post-hoc ...")
        tukey_pairs = run_tukey_hsd(groups_series, group_labels)
    else:
        print("  [INFO] ANOVA not significant. Post-hoc analysis skipped.")

    # ------------------------------------------------------------------ #
    # STEP 10 — Assemble report
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_header = [
        DIVIDER,
        HEADER,
        DIVIDER,
        f"  Test             : One-Way Analysis of Variance (ANOVA)",
        f"  Date / Time      : {timestamp}",
        f"  Outcome Variable : {outcome_col}",
        f"  Group Variable   : {group_col}",
        f"  Groups           : {', '.join(str(g) for g in group_labels)}",
        f"  Total N          : {N:,}",
        DIVIDER,
    ]

    anova_results_block = [
        "",
        "ONE-WAY ANOVA RESULTS",
        THIN_DIVIDER,
        f"  F-Statistic              : {f_stat:.4f}",
        f"  Degrees of Freedom       : df₁ = {df_between}  (between groups)",
        f"                           : df₂ = {df_within}  (within groups)",
        f"  P-Value                  : {p_value:.6f}",
        f"  Statistical Significance : {interpret_p(p_value)}",
        f"  Decision                 : {'Reject H₀ — group means are NOT all equal.' if p_value < 0.05 else 'Fail to Reject H₀ — no significant difference detected.'}",
        "",
    ]

    effect_size_block = [
        "",
        "EFFECT SIZE",
        THIN_DIVIDER,
        f"  Eta Squared (η²)  : {eta['eta_sq']:.4f}",
        f"  Magnitude         : {eta['label']}",
        f"  SS Between Groups : {eta['ss_between']:.4f}",
        f"  SS Within Groups  : {eta['ss_within']:.4f}",
        f"  SS Total          : {eta['ss_total']:.4f}",
        "  Reference: Negligible η²<0.01 | Small η²<0.06 | Medium η²<0.14 | Large η²≥0.14",
        "",
    ]

    # Assemble all sections
    report_lines = (
        report_header
        + format_descriptives_block(desc_list, outcome_col)
        + format_assumption_block(norm_results, levene_result)
        + anova_results_block
        + (format_tukey_block(tukey_pairs, outcome_col) if tukey_pairs else [
            "",
            "POST-HOC ANALYSIS",
            THIN_DIVIDER,
            "  Post-hoc analysis not performed (ANOVA was not significant).",
            "  No pairwise differences to report.",
            "",
        ])
        + effect_size_block
        + format_interpretation_block(
            f_stat, df_between, df_within, p_value,
            eta, group_labels, outcome_col, group_col,
            levene_result["equal_var"]
        )
        + [DIVIDER, "END OF REPORT", DIVIDER]
    )

    report = "\n".join(report_lines)

    # Print and save
    print("\n" + report)
    save_report(report, "anova_test_result.txt")

    return report


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Interactive entry point for the ANOVA module.

    Guides the user through:
        1. CSV file path input.
        2. Column selection.
        3. Full ANOVA analysis.
    """
    print(f"\n{DIVIDER}")
    print(f"  VM-WIT-STATS | VM Medical College")
    print(f"  Module: One-Way ANOVA")
    print(DIVIDER)

    # --- Load CSV ---
    csv_path = input("\n  Enter the full path to your CSV file: ").strip()

    try:
        df = load_csv(csv_path)
        print(f"\n  [✔] Dataset loaded successfully.  Shape: {df.shape}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    # --- Run ANOVA ---
    try:
        run_anova(df)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    print("\n  [✔] One-Way ANOVA analysis complete.")


if __name__ == "__main__":
    main()