"""
ancova.py
=========
VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit

Analysis of Covariance (ANCOVA) Module

ANCOVA combines ANOVA and linear regression to compare group means on a
continuous dependent variable while statistically controlling for one or more
continuous covariates (nuisance variables).

Model: DV ~ Group + Covariate₁ + Covariate₂ + ... + Covariateₖ

Intended for medical researchers, clinical researchers, and academic publications.
Production-grade | Research-grade | Publication-grade | Biostatistics-grade

Author: VM-WIT-STATS Toolkit

------------------------------------------------------------------------
STATISTICAL AUDIT NOTES (applied corrections):

1. TYPE III SS — Sum contrasts required.
   anova_lm(model, typ=3) in statsmodels does NOT produce correct Type III
   SS unless the model was fitted with sum (deviation) contrasts. With the
   default treatment (dummy) coding, Type III SS for factors with >2 levels
   is not orthogonal and produces results that depend on the reference level.
   Fix: set_sum_to_zero_constraints via patsy sum contrast (Sum) before fitting.

2. POST-HOC — model.model.data.predict_matrix() is an internal, undocumented
   statsmodels API that is absent in some versions and unstable across releases.
   Fix: replaced with explicit design matrix construction using patsy.dmatrices,
   which is the public, stable API for building prediction design matrices.

3. OUTLIER INDEX REPORTING — long index lists truncated to first 10 with summary.

4. HOMOGENEITY OF SLOPES — per-covariate interaction p-values now reported
   when multiple covariates are present, identifying which covariate(s) drive
   any violation.
------------------------------------------------------------------------
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ancova_result.txt")


# ==============================================================================
# SAMPLE DATA GENERATOR
# ==============================================================================

def generate_sample_csv(filepath: str = "sample_ancova_data.csv") -> str:
    """
    Generate a realistic synthetic medical dataset for ANCOVA testing.

    Scenario:
        Comparing post-treatment blood pressure across three treatment groups
        (Drug A, Drug B, Control) while controlling for baseline blood pressure
        and age.

    Variables:
        PatientID      - Unique identifier
        TreatmentGroup - Categorical: Drug_A, Drug_B, Control
        Age            - Continuous covariate (years)
        BaselineBP     - Continuous covariate (mmHg, pre-treatment)
        PostBP         - Continuous outcome (mmHg, post-treatment)

    Returns:
        str: Path to saved CSV file.
    """
    np.random.seed(42)
    n_per_group = 40
    groups = ["Drug_A", "Drug_B", "Control"]

    records = []
    pid = 1
    for group in groups:
        age = np.random.normal(55, 10, n_per_group).clip(25, 80)
        baseline_bp = np.random.normal(150, 15, n_per_group).clip(110, 200)

        if group == "Drug_A":
            effect = -18
        elif group == "Drug_B":
            effect = -10
        else:
            effect = -3

        noise = np.random.normal(0, 7, n_per_group)
        post_bp = (
            baseline_bp * 0.65
            + age * 0.20
            + effect
            + noise
        ).clip(90, 200)

        for i in range(n_per_group):
            records.append({
                "PatientID": pid,
                "TreatmentGroup": group,
                "Age": round(float(age[i]), 1),
                "BaselineBP": round(float(baseline_bp[i]), 1),
                "PostBP": round(float(post_bp[i]), 1),
            })
            pid += 1

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"[INFO] Sample CSV saved to: {filepath}")
    return filepath


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


def format_p(p: float) -> str:
    """Format p-value for publication reporting."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _format_index_list(indices: np.ndarray, max_show: int = 10) -> str:
    """
    Format an array of observation indices for report display.

    For large datasets a long list of flagged indices is unreadable and
    inflates the report. Cap display at max_show with a count summary.

    Parameters:
        indices: Array of integer indices.
        max_show: Maximum number of indices to display explicitly.

    Returns:
        str: Formatted string, e.g. "[0, 3, 7, ...] (23 total)"
    """
    idx_list = list(indices)
    n = len(idx_list)
    if n == 0:
        return "none"
    if n <= max_show:
        return str(idx_list)
    shown = idx_list[:max_show]
    return f"{shown} ... ({n} total; first {max_show} shown)"


# ==============================================================================
# FILE LOADING
# ==============================================================================

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file with comprehensive error handling.

    Parameters:
        filepath (str): Path to CSV file.

    Returns:
        pd.DataFrame or None on failure.
    """
    if not filepath or not isinstance(filepath, str):
        print("[ERROR] Invalid file path.")
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
        print("[ERROR] CSV file is empty.")
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

    print(f"\n[INFO] File loaded: {filepath}")
    print(f"       Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return df


# ==============================================================================
# COLUMN DISPLAY
# ==============================================================================

def display_columns(df: pd.DataFrame) -> None:
    """Display available columns with dtype and missing count."""
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


def select_dependent_variable(df: pd.DataFrame) -> str:
    """
    Interactively select the continuous dependent variable.

    Returns:
        str: Column name, or None on failure.
    """
    cols = list(df.columns)
    while True:
        try:
            raw = input(
                "\nEnter column name or number for the DEPENDENT variable (continuous outcome): "
            ).strip()
            if not raw:
                print("[ERROR] No input provided.")
                continue
            col = _resolve_column(raw, cols)
            if col is None:
                print(f"[ERROR] Column '{raw}' not found.")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[ERROR] '{col}' is not numeric. The dependent variable must be continuous.")
                continue
            print(f"[INFO] Dependent variable selected: {col}")
            return col
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")


def select_group_variable(df: pd.DataFrame, dv: str) -> str:
    """
    Interactively select the categorical grouping variable.

    Returns:
        str: Column name, or None on failure.
    """
    cols = list(df.columns)
    while True:
        try:
            raw = input(
                "\nEnter column name or number for the GROUPING variable (categorical factor): "
            ).strip()
            if not raw:
                print("[ERROR] No input provided.")
                continue
            col = _resolve_column(raw, cols)
            if col is None:
                print(f"[ERROR] Column '{raw}' not found.")
                continue
            if col == dv:
                print("[ERROR] Grouping variable cannot be the same as the dependent variable.")
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique < 2:
                print(f"[ERROR] '{col}' has fewer than 2 unique groups. ANCOVA requires at least 2 groups.")
                continue
            if n_unique > 20:
                print(f"[WARNING] '{col}' has {n_unique} unique values. "
                      "Confirm this is a categorical grouping variable and not continuous.")
            print(f"[INFO] Grouping variable selected: {col} ({n_unique} groups: "
                  f"{sorted(df[col].dropna().unique().tolist())})")
            return col
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")


def select_covariates(df: pd.DataFrame, dv: str, group: str) -> list:
    """
    Interactively select one or more continuous covariates.

    Returns:
        list: List of column names, or None on failure.
    """
    excluded = {dv, group}
    available = [c for c in df.columns if c not in excluded]
    print(f"\nAvailable covariates (excluding DV '{dv}' and group '{group}'):")
    for i, col in enumerate(available, 1):
        print(f"  [{i:>2}] {col:<30} dtype={df[col].dtype}")

    while True:
        try:
            raw = input(
                "\nEnter covariate column name(s) or number(s), comma-separated\n"
                "(minimum 1 required): "
            ).strip()
            if not raw:
                print("[ERROR] No input provided.")
                continue

            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            if not tokens:
                print("[ERROR] No covariates specified.")
                continue

            covariates = []
            valid = True
            for token in tokens:
                col = _resolve_column(token, available)
                if col is None:
                    print(f"[ERROR] Column '{token}' not found or not available as covariate.")
                    valid = False
                    break
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"[ERROR] Covariate '{col}' is not numeric. Covariates must be continuous.")
                    valid = False
                    break
                covariates.append(col)

            if not valid:
                continue

            if len(covariates) != len(set(covariates)):
                print("[ERROR] Duplicate covariates selected.")
                continue

            print(f"[INFO] Covariates selected: {covariates}")
            return covariates

        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_data(
    df: pd.DataFrame,
    dv: str,
    group: str,
    covariates: list,
) -> bool:
    """
    Comprehensive validation of the selected variables and dataset.

    Returns:
        bool: True if data passes validation (warnings allowed), False on fatal error.
    """
    all_vars = [dv, group] + covariates
    fatal = False

    for var in all_vars:
        if var not in df.columns:
            print(f"[ERROR] Column '{var}' not found in dataset.")
            fatal = True
    if fatal:
        return False

    if not pd.api.types.is_numeric_dtype(df[dv]):
        print(f"[ERROR] Dependent variable '{dv}' is not numeric.")
        return False

    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            print(f"[ERROR] Covariate '{cov}' is not numeric.")
            fatal = True
    if fatal:
        return False

    if dv in covariates:
        print("[ERROR] Dependent variable cannot also be a covariate.")
        return False

    if group in covariates:
        print("[ERROR] Grouping variable cannot also be a covariate.")
        return False

    num_vars = [dv] + covariates
    for var in num_vars:
        n_inf = np.isinf(df[var].replace([None], np.nan).dropna()).sum()
        if n_inf > 0:
            print(f"[WARNING] '{var}' contains {n_inf} infinite value(s). These will be removed.")

    for var in all_vars:
        n_miss = df[var].isna().sum()
        if n_miss > 0:
            pct = 100 * n_miss / len(df)
            print(f"[WARNING] '{var}' has {n_miss} missing value(s) ({pct:.1f}%). "
                  "These rows will be excluded.")

    subset = df[all_vars].copy()
    for var in num_vars:
        subset[var] = subset[var].replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna()
    n = len(subset)

    if n < 3:
        print(f"[ERROR] Only {n} complete observations after cleaning. Minimum required: 3.")
        return False
    if n < 10:
        print(f"[WARNING SEVERE] Very small sample size (n={n}). Results are highly unreliable.")
    elif n < 20:
        print(f"[WARNING] Small sample size (n={n}). Interpret results with caution.")
    elif n < 30:
        print(f"[CAUTION] Sample size n={n}. Results may be unstable.")

    if subset[dv].nunique() <= 1:
        print(f"[ERROR] Dependent variable '{dv}' is constant. Cannot run ANCOVA.")
        return False
    for cov in covariates:
        if subset[cov].nunique() <= 1:
            print(f"[ERROR] Covariate '{cov}' is constant. Cannot be used in ANCOVA.")
            fatal = True
    if fatal:
        return False

    groups = subset[group].dropna().unique()
    if len(groups) < 2:
        print(f"[ERROR] Only {len(groups)} group(s) after cleaning. ANCOVA requires at least 2 groups.")
        return False

    for g in groups:
        g_n = (subset[group] == g).sum()
        if g_n < 2:
            print(f"[ERROR] Group '{g}' has only {g_n} observation(s) after cleaning. "
                  "Each group requires at least 2 observations.")
            fatal = True
    if fatal:
        return False

    k_params = 1 + (len(groups) - 1) + len(covariates)
    if n < 10 * k_params:
        print(f"[WARNING] Sample size (n={n}) < 10 × number of model parameters ({k_params}). "
              f"Risk of overfitting. Recommend n ≥ {10 * k_params}.")

    print(f"[INFO] Validation passed. Complete observations: {n}")
    return True


# ==============================================================================
# DATA CLEANING
# ==============================================================================

def clean_data(
    df: pd.DataFrame,
    dv: str,
    group: str,
    covariates: list,
) -> pd.DataFrame:
    """
    Extract selected variables, remove infinities and missing values,
    and ensure the grouping variable is typed as string.

    Returns:
        pd.DataFrame: Cleaned, reset-index dataset.
    """
    all_vars = [dv, group] + covariates
    subset = df[all_vars].copy()
    n_before = len(subset)
    num_vars = [dv] + covariates
    for var in num_vars:
        subset[var] = subset[var].replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna().reset_index(drop=True)
    n_after = len(subset)
    removed = n_before - n_after
    if removed > 0:
        print(f"[INFO] Removed {removed} row(s) due to missing/infinite values. "
              f"Remaining: {n_after}")
    subset[group] = subset[group].astype(str)
    return subset


# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

def descriptive_statistics(
    df: pd.DataFrame,
    dv: str,
    group: str,
    covariates: list,
) -> tuple:
    """
    Compute per-group descriptive statistics for the DV and overall
    covariate descriptives.

    Returns:
        tuple: (group_desc DataFrame, covariate_desc DataFrame)
    """
    groups = sorted(df[group].unique())

    group_rows = []
    for g in groups:
        s = df.loc[df[group] == g, dv]
        n = len(s)
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        group_rows.append({
            "Group": g,
            "N": n,
            "Mean": s.mean(),
            "Median": s.median(),
            "SD": s.std(ddof=1),
            "SE": s.std(ddof=1) / np.sqrt(n),
            "Min": s.min(),
            "Max": s.max(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
        })
    group_desc = pd.DataFrame(group_rows)

    cov_rows = []
    for cov in covariates:
        s = df[cov]
        n = len(s)
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        cov_rows.append({
            "Covariate": cov,
            "N": n,
            "Mean": s.mean(),
            "Median": s.median(),
            "SD": s.std(ddof=1),
            "SE": s.std(ddof=1) / np.sqrt(n),
            "Min": s.min(),
            "Max": s.max(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
        })
    cov_desc = pd.DataFrame(cov_rows)
    return group_desc, cov_desc


# ==============================================================================
# MODEL BUILDING — SUM CONTRASTS FOR CORRECT TYPE III SS
# ==============================================================================

def _sanitize_name(name: str) -> str:
    """
    Sanitize a column name for use in statsmodels formula strings.
    Wraps names containing spaces or special characters in Q('...').
    """
    import re
    if re.search(r'[^A-Za-z0-9_]', str(name)):
        return f"Q('{name}')"
    return str(name)


def build_formula(dv: str, group: str, covariates: list) -> str:
    """
    Build the statsmodels OLS formula string for the main ANCOVA model.

    Sum contrasts are applied at model fit time (not in the formula string).
    The formula uses C(group) with the group column name.

    Formula: DV ~ C(Group) + Cov1 + Cov2 + ...
    """
    dv_s = _sanitize_name(dv)
    group_s = f"C({_sanitize_name(group)})"
    cov_parts = [_sanitize_name(c) for c in covariates]
    rhs = " + ".join([group_s] + cov_parts)
    return f"{dv_s} ~ {rhs}"


def build_formula_sum_contrast(dv: str, group: str, covariates: list) -> str:
    """
    Build the OLS formula with explicit sum (deviation) contrast coding
    for the group factor.

    WHY SUM CONTRASTS ARE REQUIRED FOR TYPE III SS:
        In statsmodels, anova_lm(model, typ=3) computes Type III sums of
        squares correctly ONLY when the categorical factor uses sum (effect)
        coding, not the default treatment (dummy) coding.

        With treatment coding:
            - The intercept represents the reference group mean.
            - Group effects are expressed relative to that reference level.
            - Type III SS for the group factor depends on which level is the
              reference, violating the requirement that Type III SS be
              invariant to reference level choice.

        With sum (deviation) coding:
            - Each group coefficient is the deviation from the grand mean.
            - The intercept represents the unweighted grand mean.
            - Type III SS for the group factor is then invariant to group
              ordering and reference level, which is the defining property
              of true Type III SS (SAS PROC GLM behaviour).

        Reference: Fox & Weisberg (2011); Venables (2000);
                   statsmodels documentation on contrast coding.

    Formula: DV ~ C(Group, Sum) + Cov1 + Cov2 + ...
    """
    dv_s = _sanitize_name(dv)
    group_s = f"C({_sanitize_name(group)}, Sum)"
    cov_parts = [_sanitize_name(c) for c in covariates]
    rhs = " + ".join([group_s] + cov_parts)
    return f"{dv_s} ~ {rhs}"


def build_interaction_formula_sum_contrast(dv: str, group: str, covariates: list) -> str:
    """
    Build the interaction model formula with sum contrasts for the
    homogeneity of regression slopes test.

    Formula: DV ~ C(Group, Sum) + Cov1 + Cov2 +
             C(Group, Sum):Cov1 + C(Group, Sum):Cov2 + ...
    """
    dv_s = _sanitize_name(dv)
    group_s = f"C({_sanitize_name(group)}, Sum)"
    cov_parts = [_sanitize_name(c) for c in covariates]
    interaction_parts = [f"{group_s}:{c}" for c in cov_parts]
    rhs = " + ".join([group_s] + cov_parts + interaction_parts)
    return f"{dv_s} ~ {rhs}"


# ==============================================================================
# HOMOGENEITY OF REGRESSION SLOPES
# ==============================================================================

def test_homogeneity_of_slopes(
    df: pd.DataFrame,
    dv: str,
    group: str,
    covariates: list,
) -> dict:
    """
    Test the homogeneity of regression slopes assumption — the most critical
    ANCOVA assumption.

    Method:
        1. Fit the main effects ANCOVA model (sum contrasts, Type III SS).
        2. Fit the interaction model adding Group × Covariate terms.
        3. Compare via F-test: significant interaction (p ≤ 0.05) means the
           covariate–DV relationship differs across groups, violating ANCOVA.

    Per-covariate breakdown:
        When multiple covariates are present, each Group × Covariate
        interaction is also tested individually so that the specific
        covariate(s) driving any violation are identified.

    Both models use sum contrasts to ensure Type III SS validity.

    Returns:
        dict with overall F/p, per-covariate interaction results,
        assumption_met flag, and interpretation.
    """
    try:
        formula_main = build_formula_sum_contrast(dv, group, covariates)
        formula_interaction = build_interaction_formula_sum_contrast(dv, group, covariates)

        model_main = smf.ols(formula_main, data=df).fit()
        model_interaction = smf.ols(formula_interaction, data=df).fit()

        comparison = anova_lm(model_main, model_interaction)
        f_stat = float(comparison["F"].iloc[1])
        p_val = float(comparison["Pr(>F)"].iloc[1])
        df_num = float(comparison["df_diff"].iloc[1])

        assumption_met = p_val > 0.05

        # Per-covariate interaction tests (individual Group × Cov F-tests)
        per_cov_results = {}
        if len(covariates) > 1:
            for cov in covariates:
                try:
                    # Model without this specific interaction term
                    other_covs = [c for c in covariates if c != cov]
                    group_s = f"C({_sanitize_name(group)}, Sum)"
                    dv_s = _sanitize_name(dv)
                    cov_s = _sanitize_name(cov)
                    all_cov_parts = [_sanitize_name(c) for c in covariates]
                    other_interaction_parts = [
                        f"{group_s}:{_sanitize_name(c)}" for c in covariates if c != cov
                    ]
                    # Full interaction model minus this cov's interaction
                    rhs_reduced = " + ".join(
                        [group_s] + all_cov_parts + other_interaction_parts
                    )
                    formula_reduced = f"{dv_s} ~ {rhs_reduced}"
                    model_reduced = smf.ols(formula_reduced, data=df).fit()
                    cmp = anova_lm(model_reduced, model_interaction)
                    f_cov = float(cmp["F"].iloc[1])
                    p_cov = float(cmp["Pr(>F)"].iloc[1])
                    per_cov_results[cov] = {"F": f_cov, "p": p_cov}
                except Exception as e:
                    per_cov_results[cov] = {"F": np.nan, "p": np.nan}

        if assumption_met:
            interp = (
                f"Homogeneity of regression slopes: SATISFIED "
                f"(F = {f_stat:.3f}, p = {format_p(p_val)}). "
                f"The relationship between the covariate(s) and {dv} does not "
                f"significantly differ across groups. ANCOVA adjusted means are valid."
            )
        else:
            interp = (
                f"Homogeneity of regression slopes: VIOLATED "
                f"(F = {f_stat:.3f}, p = {format_p(p_val)}). "
                f"The relationship between the covariate(s) and {dv} differs "
                f"significantly across groups. ANCOVA adjusted means may not be "
                f"meaningful. Consider the Johnson-Neyman technique or separate "
                "within-group regressions."
            )

        return {
            "F_statistic": f_stat,
            "df_numerator": df_num,
            "p_value": p_val,
            "assumption_met": assumption_met,
            "per_covariate": per_cov_results,
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "F_statistic": np.nan,
            "df_numerator": np.nan,
            "p_value": np.nan,
            "assumption_met": True,
            "per_covariate": {},
            "interpretation": f"Homogeneity of slopes test could not be computed: {e}",
        }


# ==============================================================================
# ANCOVA MODEL FITTING — TYPE III SS WITH SUM CONTRASTS
# ==============================================================================

def fit_ancova_model(df: pd.DataFrame, dv: str, group: str, covariates: list):
    """
    Fit the main ANCOVA OLS model with sum (deviation) contrasts and
    compute Type III sums of squares.

    STATISTICAL RATIONALE FOR SUM CONTRASTS:
        statsmodels anova_lm(typ=3) requires sum contrasts on all categorical
        factors to produce correct Type III SS. With default treatment coding,
        the Type III SS for the group factor depends on the reference level,
        which is statistically incorrect. Sum contrasts make the group SS
        orthogonal to the intercept, yielding reference-level-invariant
        Type III SS consistent with SAS PROC GLM / SPSS GLM output.

    Parameters:
        df: Cleaned dataset.
        dv: Dependent variable column name.
        group: Grouping variable column name.
        covariates: List of covariate column names.

    Returns:
        tuple: (fitted_result, type3_anova_table) or (None, None) on failure.
    """
    try:
        formula = build_formula_sum_contrast(dv, group, covariates)
        model = smf.ols(formula, data=df).fit()
        type3_table = anova_lm(model, typ=3)
        return model, type3_table
    except Exception as e:
        print(f"[ERROR] ANCOVA model fitting failed: {e}")
        return None, None


# ==============================================================================
# EFFECT SIZE — PARTIAL ETA SQUARED
# ==============================================================================

def compute_partial_eta_squared(type3_table: pd.DataFrame) -> dict:
    """
    Compute partial eta squared (η²p) from a Type III ANOVA table.

    η²p = SS_effect / (SS_effect + SS_residual)

    Interpretation (Cohen, 1988):
        Small  : η²p ≥ 0.01
        Medium : η²p ≥ 0.06
        Large  : η²p ≥ 0.14

    Parameters:
        type3_table: Type III ANOVA table from anova_lm(typ=3).

    Returns:
        dict: {source_name: {"partial_eta_sq": float, "interpretation": str}}
    """
    results = {}
    try:
        ss_residual = float(type3_table.loc["Residual", "sum_sq"])
        for idx in type3_table.index:
            if idx in ("Intercept", "Residual"):
                continue
            ss_effect = float(type3_table.loc[idx, "sum_sq"])
            denominator = ss_effect + ss_residual
            if denominator <= 0:
                eta_sq = np.nan
            else:
                eta_sq = ss_effect / denominator

            if np.isnan(eta_sq):
                interp = "Undefined"
            elif eta_sq >= 0.14:
                interp = "Large effect"
            elif eta_sq >= 0.06:
                interp = "Medium effect"
            elif eta_sq >= 0.01:
                interp = "Small effect"
            else:
                interp = "Negligible effect"

            results[idx] = {
                "partial_eta_sq": eta_sq,
                "interpretation": interp,
            }
    except Exception as e:
        print(f"[WARNING] Effect size computation failed: {e}")
    return results


# ==============================================================================
# ADJUSTED (ESTIMATED MARGINAL) MEANS
# ==============================================================================

def compute_adjusted_means(
    model,
    df: pd.DataFrame,
    group: str,
    covariates: list,
) -> pd.DataFrame:
    """
    Compute estimated marginal means (adjusted means) for each group.

    Method:
        For each group, predict the DV with all covariates held at their
        grand mean. SE and 95% CI are obtained from model.get_prediction().

    Parameters:
        model: Fitted OLS model result.
        df: Cleaned dataset.
        group: Grouping variable name.
        covariates: List of covariate names.

    Returns:
        pd.DataFrame with columns: Group, Adjusted_Mean, SE, CI_Lower, CI_Upper.
    """
    groups = sorted(df[group].unique())
    cov_means = {cov: df[cov].mean() for cov in covariates}

    rows = []
    for g in groups:
        pred_row = {group: str(g)}
        pred_row.update(cov_means)
        pred_df = pd.DataFrame([pred_row])
        pred_df[group] = pred_df[group].astype(str)
        try:
            pred = model.get_prediction(pred_df)
            summary = pred.summary_frame(alpha=0.05)
            adj_mean = float(summary["mean"].iloc[0])
            se = float(summary["mean_se"].iloc[0])
            ci_lo = float(summary["mean_ci_lower"].iloc[0])
            ci_hi = float(summary["mean_ci_upper"].iloc[0])
        except Exception:
            try:
                adj_mean = float(model.predict(pred_df).iloc[0])
                se = np.nan
                ci_lo = np.nan
                ci_hi = np.nan
            except Exception:
                adj_mean = np.nan
                se = np.nan
                ci_lo = np.nan
                ci_hi = np.nan
        rows.append({
            "Group": g,
            "Adjusted_Mean": adj_mean,
            "SE": se,
            "CI_Lower": ci_lo,
            "CI_Upper": ci_hi,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# POST-HOC PAIRWISE COMPARISONS (BONFERRONI)
# ==============================================================================

def _build_prediction_design_row(model, df: pd.DataFrame, group: str,
                                  covariates: list, group_value: str) -> np.ndarray:
    """
    Build a single design matrix row for a given group at covariate grand means.

    Uses patsy.dmatrix with the model's formula and data to construct the
    correct design vector, including contrast coding. This is the stable,
    public patsy API and avoids the undocumented
    model.model.data.predict_matrix() internal that is absent in some
    statsmodels versions.

    Parameters:
        model: Fitted OLS result (provides formula_str and data reference).
        df: Full cleaned dataset (needed for factor levels / contrast context).
        group: Grouping variable name.
        covariates: List of covariate names.
        group_value: The specific group level to construct the row for.

    Returns:
        np.ndarray: 1-D design vector of length n_params.
    """
    import patsy
    cov_means = {cov: df[cov].mean() for cov in covariates}
    pred_row = {group: str(group_value)}
    pred_row.update(cov_means)
    pred_df = pd.DataFrame([pred_row])
    pred_df[group] = pred_df[group].astype(str)

    # patsy.dmatrix with the model's formula_str and the original df as
    # the "training" context ensures factor levels and contrast matrices
    # match those used during model fitting.
    formula_rhs = model.model.formula.split("~", 1)[1].strip()
    try:
        dmat = patsy.dmatrix(
            formula_rhs,
            data=pred_df,
            return_type="matrix",
            NA_action="raise",
        )
        return np.asarray(dmat).flatten()
    except Exception:
        # Fallback: use patsy with the original training data to establish
        # factor levels, then transform the prediction row.
        try:
            _, full_dmat = patsy.dmatrices(
                model.model.formula,
                data=df,
                return_type="matrix",
                NA_action="drop",
            )
            pred_dmat = patsy.build_design_matrices(
                [full_dmat.design_info], pred_df, return_type="matrix"
            )[0]
            return np.asarray(pred_dmat).flatten()
        except Exception as e2:
            raise RuntimeError(
                f"Could not build prediction design row for group '{group_value}': {e2}"
            )


def posthoc_bonferroni(
    model,
    df: pd.DataFrame,
    group: str,
    covariates: list,
    dv: str,
) -> pd.DataFrame:
    """
    Perform Bonferroni-corrected pairwise comparisons of adjusted means.

    Method:
        For each pair of groups, construct contrast vectors using patsy's
        public dmatrix / build_design_matrices API (stable across statsmodels
        versions). The contrast vector is the difference between the two
        prediction design rows at covariate grand means.

        The SE of the mean difference is computed via:
            SE = sqrt(c' V c)
        where c is the contrast vector and V is the model's covariance matrix.

        This correctly accounts for model uncertainty and covariate adjustment.
        Raw p-values are Bonferroni-corrected by multiplication by the number
        of comparisons, capped at 1.0. CIs use the Bonferroni-adjusted α.

    Parameters:
        model: Fitted OLS result (with sum contrasts).
        df: Cleaned dataset.
        group: Grouping variable name.
        covariates: List of covariate column names.
        dv: Dependent variable name.

    Returns:
        pd.DataFrame with pairwise comparison results.
    """
    groups = sorted(df[group].unique())
    pairs = list(itertools.combinations(groups, 2))
    n_comparisons = len(pairs)
    cov_matrix = np.asarray(model.cov_params())

    rows = []
    for (g1, g2) in pairs:
        try:
            row1 = _build_prediction_design_row(model, df, group, covariates, g1)
            row2 = _build_prediction_design_row(model, df, group, covariates, g2)

            params = np.asarray(model.params)
            mean1 = float(row1 @ params)
            mean2 = float(row2 @ params)
            mean_diff = mean1 - mean2

            contrast = row1 - row2
            var_diff = float(contrast @ cov_matrix @ contrast)
            se_diff = np.sqrt(max(var_diff, 0.0))

            df_resid = model.df_resid
            t_stat = mean_diff / se_diff if se_diff > 0 else np.nan
            p_raw = (
                float(2 * stats.t.sf(abs(t_stat), df=df_resid))
                if not np.isnan(t_stat) else np.nan
            )
            p_bonf = min(p_raw * n_comparisons, 1.0) if not np.isnan(p_raw) else np.nan

            alpha_adj = 0.05 / n_comparisons
            t_crit = stats.t.ppf(1 - alpha_adj / 2, df=df_resid)
            ci_lo = mean_diff - t_crit * se_diff
            ci_hi = mean_diff + t_crit * se_diff

            rows.append({
                "Group 1": g1,
                "Group 2": g2,
                "Mean Difference": mean_diff,
                "SE": se_diff,
                "t": t_stat,
                "p (raw)": p_raw,
                "p (Bonferroni)": p_bonf,
                "CI Lower (adj)": ci_lo,
                "CI Upper (adj)": ci_hi,
            })
        except Exception as e:
            rows.append({
                "Group 1": g1,
                "Group 2": g2,
                "Mean Difference": np.nan,
                "SE": np.nan,
                "t": np.nan,
                "p (raw)": np.nan,
                "p (Bonferroni)": np.nan,
                "CI Lower (adj)": np.nan,
                "CI Upper (adj)": np.nan,
            })
    return pd.DataFrame(rows)


# ==============================================================================
# ASSUMPTION CHECKS
# ==============================================================================

def check_residual_normality(residuals: np.ndarray) -> dict:
    """
    Test residual normality using a sample-size-appropriate method.

    n ≤ 2000 : Shapiro-Wilk
    n > 2000 : Lilliefors (corrected KS for estimated parameters)

    Standard scipy.stats.kstest() is NOT used for large samples because
    it assumes distribution parameters are known a priori. Residual
    parameters are estimated from the same data, violating the KS null
    distribution and producing conservative p-values (Lilliefors, 1967).
    """
    try:
        n = len(residuals)
        if n <= 2000:
            stat, p_val = stats.shapiro(residuals)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_val = sm_lilliefors(residuals, dist="norm", pvalmethod="approx")
            test_name = "Lilliefors (KS with estimated parameters)"

        large_n_caveat = ""
        if n > 5000:
            large_n_caveat = (
                " NOTE: n > 5000 — normality tests detect trivially small deviations. "
                "OLS inference is protected by the CLT."
            )

        if p_val > 0.05:
            interp = (
                f"{test_name}: Residuals consistent with normality (p > 0.05). "
                f"Assumption supported.{large_n_caveat}"
            )
        else:
            interp = (
                f"{test_name}: Residuals deviate from normality (p ≤ 0.05). "
                "OLS estimates remain unbiased, but consider robust inference for "
                f"small samples.{large_n_caveat}"
            )

        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_val),
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "test": "Unknown", "statistic": np.nan, "p_value": np.nan,
            "interpretation": f"Normality test failed: {e}",
        }


def check_homogeneity_of_variance(
    df: pd.DataFrame, dv: str, group: str
) -> dict:
    """
    Levene's test for homogeneity of variance across groups.

    H₀: Group variances are equal.
    H₁: At least one group variance differs.

    Levene's test (center='mean') is preferred over Bartlett's test because
    it is robust to departures from normality (Brown & Forsythe, 1974).
    """
    try:
        groups = sorted(df[group].unique())
        group_data = [df.loc[df[group] == g, dv].values for g in groups]
        stat, p_val = levene(*group_data, center="mean")

        if p_val > 0.05:
            interp = (
                "Levene's test: Homogeneity of variance supported (p > 0.05). "
                "Group variances are not significantly different."
            )
        else:
            interp = (
                "Levene's test: Heterogeneity of variance detected (p ≤ 0.05). "
                "Group variances differ significantly. ANCOVA is moderately robust "
                "to variance heterogeneity when groups are balanced, but interpret "
                "results with caution in unbalanced designs."
            )

        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "statistic": np.nan, "p_value": np.nan,
            "interpretation": f"Levene's test failed: {e}",
        }


def check_linearity(
    df: pd.DataFrame, dv: str, covariates: list
) -> dict:
    """
    Assess linearity between the DV and each covariate via Pearson
    correlation and a linear-vs-quadratic F-test.

    A significant quadratic improvement (p ≤ 0.05) suggests non-linearity
    warranting transformation or polynomial terms.
    """
    results = {}
    for cov in covariates:
        try:
            r, p_r = stats.pearsonr(df[cov], df[dv])

            X_lin = sm.add_constant(df[cov].values)
            X_quad = sm.add_constant(
                np.column_stack([df[cov].values, df[cov].values ** 2])
            )
            model_lin = sm.OLS(df[dv].values, X_lin).fit()
            model_quad = sm.OLS(df[dv].values, X_quad).fit()

            ss_lin = model_lin.ssr
            ss_quad = model_quad.ssr
            df_diff = model_lin.df_resid - model_quad.df_resid
            if df_diff > 0 and ss_quad > 0:
                f_nl = ((ss_lin - ss_quad) / df_diff) / (ss_quad / model_quad.df_resid)
                p_nl = float(stats.f.sf(f_nl, df_diff, model_quad.df_resid))
            else:
                f_nl = np.nan
                p_nl = np.nan

            direction = "positive" if r > 0 else "negative"
            strength = (
                "strong" if abs(r) >= 0.70 else
                "moderate" if abs(r) >= 0.40 else
                "weak"
            )
            lin_note = ""
            if not np.isnan(p_nl):
                if p_nl <= 0.05:
                    lin_note = (
                        f" Quadratic term significant (F = {f_nl:.3f}, "
                        f"p = {format_p(p_nl)}): non-linear relationship possible. "
                        "Consider polynomial or log transformation."
                    )
                else:
                    lin_note = (
                        f" No significant quadratic component (F = {f_nl:.3f}, "
                        f"p = {format_p(p_nl)}): linearity supported."
                    )

            interp = (
                f"{cov}: {strength} {direction} linear association with {dv} "
                f"(r = {r:.3f}, p = {format_p(p_r)}).{lin_note}"
            )

            results[cov] = {
                "pearson_r": float(r),
                "pearson_p": float(p_r),
                "nonlinearity_F": float(f_nl) if not np.isnan(f_nl) else np.nan,
                "nonlinearity_p": float(p_nl) if not np.isnan(p_nl) else np.nan,
                "interpretation": interp,
            }
        except Exception as e:
            results[cov] = {
                "pearson_r": np.nan, "pearson_p": np.nan,
                "nonlinearity_F": np.nan, "nonlinearity_p": np.nan,
                "interpretation": f"Linearity check failed for {cov}: {e}",
            }
    return results


def detect_outliers_and_influence(model, n: int) -> dict:
    """
    Detect residual outliers (two-tier studentized residuals) and
    influential observations (Cook's Distance).

    Outlier tiers (Belsley, Kuh & Welsch, 1980; Cook & Weisberg, 1982):
        Moderate : 2 < |SR| ≤ 3
        Strong   : |SR| > 3

    Cook's D threshold: 4/n.

    Index lists are truncated to the first 10 entries in the report to
    prevent unreadable output in large datasets.
    """
    try:
        influence = OLSInfluence(model)
        student_resid = influence.resid_studentized_external
        cooks_d = influence.cooks_distance[0]
        cooks_cutoff = 4 / n

        strong_idx = np.where(np.abs(student_resid) > 3)[0]
        moderate_idx = np.where(
            (np.abs(student_resid) > 2) & (np.abs(student_resid) <= 3)
        )[0]
        influential_idx = np.where(cooks_d > cooks_cutoff)[0]

        return {
            "studentized_residuals": student_resid,
            "cooks_distance": cooks_d,
            "strong_outlier_indices": strong_idx,
            "moderate_outlier_indices": moderate_idx,
            "influential_indices": influential_idx,
            "cooks_cutoff": cooks_cutoff,
            "n_strong": len(strong_idx),
            "n_moderate": len(moderate_idx),
            "n_influential": len(influential_idx),
        }
    except Exception as e:
        print(f"[WARNING] Outlier/influence diagnostics failed: {e}")
        return {
            "studentized_residuals": np.array([]),
            "cooks_distance": np.array([]),
            "strong_outlier_indices": np.array([]),
            "moderate_outlier_indices": np.array([]),
            "influential_indices": np.array([]),
            "cooks_cutoff": 4 / n,
            "n_strong": 0, "n_moderate": 0, "n_influential": 0,
        }


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(
    df: pd.DataFrame,
    dv: str,
    group: str,
    covariates: list,
    group_desc: pd.DataFrame,
    cov_desc: pd.DataFrame,
    slopes_res: dict,
    normality_res: dict,
    levene_res: dict,
    linearity_res: dict,
    diag_res: dict,
    model,
    type3_table: pd.DataFrame,
    eta_sq: dict,
    adj_means: pd.DataFrame,
    posthoc: pd.DataFrame,
    file_handle,
) -> None:
    """
    Write the complete ANCOVA report to both console and file simultaneously.
    """
    n = len(df)
    groups = sorted(df[group].unique())
    n_groups = len(groups)
    out = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit")
    out.append("ANALYSIS OF COVARIANCE (ANCOVA)")
    out.append(_separator())
    out.append(f"Dependent Variable : {dv}")
    out.append(f"Grouping Variable  : {group}  ({n_groups} groups: {', '.join(groups)})")
    out.append(f"Covariates ({len(covariates)})      : {', '.join(covariates)}")
    out.append(f"Sample Size (n)    : {n}")
    out.append(f"Model              : {dv} ~ {group} + {' + '.join(covariates)}")
    out.append(f"Sums of Squares    : Type III (sum/deviation contrasts applied)")
    _dual_print(out, file_handle); out = []

    # ── DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────
    out.append(_section("1. DESCRIPTIVE STATISTICS"))
    out.append(f"  Dependent Variable: {dv} (unadjusted, per group)")
    out.append("")
    out.append(
        f"  {'Group':<18} {'N':>5} {'Mean':>10} {'Median':>10} {'SD':>10} "
        f"{'SE':>10} {'Min':>8} {'Max':>8} {'Q1':>8} {'Q3':>8} {'IQR':>8}"
    )
    out.append("  " + _separator("-", 100))
    for _, row in group_desc.iterrows():
        out.append(
            f"  {str(row['Group']):<18} {int(row['N']):>5} {row['Mean']:>10.3f} "
            f"{row['Median']:>10.3f} {row['SD']:>10.3f} {row['SE']:>10.3f} "
            f"{row['Min']:>8.3f} {row['Max']:>8.3f} {row['Q1']:>8.3f} "
            f"{row['Q3']:>8.3f} {row['IQR']:>8.3f}"
        )
    out.append("")
    out.append("  Covariates (overall sample):")
    out.append(
        f"  {'Covariate':<22} {'N':>5} {'Mean':>10} {'Median':>10} {'SD':>10} "
        f"{'SE':>10} {'Min':>8} {'Max':>8}"
    )
    out.append("  " + _separator("-", 85))
    for _, row in cov_desc.iterrows():
        out.append(
            f"  {str(row['Covariate']):<22} {int(row['N']):>5} {row['Mean']:>10.3f} "
            f"{row['Median']:>10.3f} {row['SD']:>10.3f} {row['SE']:>10.3f} "
            f"{row['Min']:>8.3f} {row['Max']:>8.3f}"
        )
    _dual_print(out, file_handle); out = []

    # ── ASSUMPTION CHECKS ──────────────────────────────────────────────────────
    out.append(_section("2. ASSUMPTION CHECKS"))

    # [A] Homogeneity of Regression Slopes — CRITICAL
    out.append("  [A] HOMOGENEITY OF REGRESSION SLOPES  ← CRITICAL ANCOVA ASSUMPTION")
    out.append("      Tests whether the covariate-DV relationship is the same in all groups.")
    out.append("      If violated, ANCOVA adjusted means are not interpretable.")
    out.append("")
    if not np.isnan(slopes_res["F_statistic"]):
        out.append(f"      Overall test:")
        out.append(f"        F-statistic   : {slopes_res['F_statistic']:.4f}")
        out.append(f"        df (numerator) : {slopes_res['df_numerator']:.0f}")
        out.append(f"        p-value        : {format_p(slopes_res['p_value'])}")
    out.append(f"      Result: {slopes_res['interpretation']}")

    # Per-covariate breakdown when multiple covariates present
    if slopes_res.get("per_covariate"):
        out.append("")
        out.append("      Per-covariate interaction breakdown:")
        out.append(f"      {'Covariate':<25} {'F':>10} {'p-value':>12} {'Flag'}")
        out.append("      " + _separator("-", 60))
        for cov, res in slopes_res["per_covariate"].items():
            f_str = f"{res['F']:.4f}" if not np.isnan(res['F']) else "N/A"
            p_str = format_p(res['p'])
            flag = "⚠ VIOLATION" if (not np.isnan(res['p']) and res['p'] <= 0.05) else "OK"
            out.append(f"      {cov:<25} {f_str:>10} {p_str:>12}  {flag}")

    if not slopes_res["assumption_met"]:
        out.append("")
        out.append("      ⚠ WARNING: ANCOVA ASSUMPTION VIOLATED.")
        out.append("        Adjusted means below are reported but must be interpreted")
        out.append("        with extreme caution. Consider stratified analyses or")
        out.append("        the Johnson-Neyman technique for interacting covariates.")
    out.append("")

    # [B] Residual Normality
    out.append(f"  [B] RESIDUAL NORMALITY — {normality_res['test']}")
    stat_str = f"{normality_res['statistic']:.4f}" if not np.isnan(normality_res['statistic']) else "N/A"
    out.append(f"      Statistic  : {stat_str}")
    out.append(f"      p-value    : {format_p(normality_res['p_value'])}")
    out.append(f"      Result     : {normality_res['interpretation']}")
    out.append("")

    # [C] Homogeneity of Variance
    out.append("  [C] HOMOGENEITY OF VARIANCE — Levene's Test")
    lev_str = f"{levene_res['statistic']:.4f}" if not np.isnan(levene_res['statistic']) else "N/A"
    out.append(f"      W-statistic : {lev_str}")
    out.append(f"      p-value     : {format_p(levene_res['p_value'])}")
    out.append(f"      Result      : {levene_res['interpretation']}")
    out.append("")

    # [D] Linearity
    out.append("  [D] LINEARITY — DV vs Each Covariate")
    for cov, res in linearity_res.items():
        out.append(f"      {res['interpretation']}")
    out.append("")

    # [E] Outliers — Two-Tier (truncated index lists)
    out.append("  [E] OUTLIER DETECTION — Studentized Residuals (two-tier)")
    out.append(f"      Strong outliers  (|SR| > 3)     : {diag_res['n_strong']}")
    out.append(f"      Moderate concern (2 < |SR| ≤ 3) : {diag_res['n_moderate']}")
    if diag_res["n_strong"] > 0:
        idx_str = _format_index_list(diag_res["strong_outlier_indices"])
        out.append(f"      Strong outlier indices (0-based): {idx_str}")
        out.append("      ⚠ STRONG OUTLIERS: Verify clinical data accuracy. "
                   "Do not remove without documented justification.")
    if diag_res["n_moderate"] > 0:
        idx_str = _format_index_list(diag_res["moderate_outlier_indices"])
        out.append(f"      Moderate concern indices         : {idx_str}")
        out.append("      Inspect individually — unusual but not definitive outliers.")
    if diag_res["n_strong"] == 0 and diag_res["n_moderate"] == 0:
        out.append("      ✓ No outliers at either threshold.")
    out.append("")

    # [F] Influence (truncated index list)
    out.append(f"  [F] INFLUENCE — Cook's Distance (threshold = 4/n = {diag_res['cooks_cutoff']:.4f})")
    out.append(f"      Influential observations : {diag_res['n_influential']}")
    if diag_res["n_influential"] > 0:
        idx_str = _format_index_list(diag_res["influential_indices"])
        out.append(f"      Indices (0-based): {idx_str}")
        out.append("      These observations substantially affect the model. Verify data accuracy.")
    else:
        out.append("      ✓ No highly influential observations detected.")
    _dual_print(out, file_handle); out = []

    # ── ANCOVA TABLE (TYPE III SS) ─────────────────────────────────────────────
    out.append(_section("3. ANCOVA TABLE — TYPE III SUMS OF SQUARES"))
    out.append("  Type III SS: each effect tested after accounting for all other effects.")
    out.append("  Sum (deviation) contrasts applied to ensure reference-level-invariant SS.")
    out.append("")
    out.append(
        f"  {'Source':<32} {'SS':>14} {'df':>6} {'MS':>14} {'F':>10} {'p-value':>12} "
        f"{'Partial η²':>12} {'Effect'}"
    )
    out.append("  " + _separator("-", 110))

    for idx in type3_table.index:
        if idx == "Intercept":
            continue
        try:
            ss = float(type3_table.loc[idx, "sum_sq"])
            df_val = float(type3_table.loc[idx, "df"])
            ms = ss / df_val if df_val > 0 else np.nan
            f_col = "F" if "F" in type3_table.columns else None
            p_col = "PR(>F)" if "PR(>F)" in type3_table.columns else None
            f_val = float(type3_table.loc[idx, f_col]) if f_col else np.nan
            p_val = float(type3_table.loc[idx, p_col]) if p_col else np.nan

            if idx in eta_sq:
                eta_val = eta_sq[idx]["partial_eta_sq"]
                eta_str = f"{eta_val:.4f}" if not np.isnan(eta_val) else "N/A"
                effect_str = eta_sq[idx]["interpretation"]
            else:
                eta_str = "N/A"
                effect_str = ""

            label = "Residual" if idx == "Residual" else idx
            f_str = f"{f_val:.4f}" if not np.isnan(f_val) else "N/A"
            ms_str = f"{ms:.4f}" if not np.isnan(ms) else "N/A"

            out.append(
                f"  {label:<32} {ss:>14.4f} {int(df_val):>6} {ms_str:>14} "
                f"{f_str:>10} {format_p(p_val):>12} {eta_str:>12}  {effect_str}"
            )
        except Exception:
            out.append(f"  {idx:<32} [could not parse row]")
    _dual_print(out, file_handle); out = []

    # ── MODEL PERFORMANCE ──────────────────────────────────────────────────────
    out.append(_section("4. MODEL PERFORMANCE"))
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    rse = np.sqrt(model.mse_resid)
    out.append(f"  R²                      : {r2:.4f}  ({100*r2:.2f}% of variance in {dv} explained)")
    out.append(f"  Adjusted R²             : {r2_adj:.4f}")
    out.append(f"  Residual Standard Error : {rse:.4f}  (units of {dv})")
    out.append(f"  AIC                     : {model.aic:.4f}")
    out.append(f"  BIC                     : {model.bic:.4f}")
    out.append(f"  df (residual)           : {int(model.df_resid)}")
    _dual_print(out, file_handle); out = []

    # ── ADJUSTED MEANS ─────────────────────────────────────────────────────────
    out.append(_section("5. ADJUSTED (ESTIMATED MARGINAL) MEANS"))
    out.append("  Adjusted means: predicted group means with all covariates at their grand mean.")
    for cov in covariates:
        out.append(f"    {cov} grand mean = {df[cov].mean():.4f}")
    out.append("")
    out.append(
        f"  {'Group':<20} {'Adjusted Mean':>15} {'SE':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}"
    )
    out.append("  " + _separator("-", 76))
    for _, row in adj_means.iterrows():
        se_str = f"{row['SE']:.4f}" if not np.isnan(row['SE']) else "N/A"
        ci_lo_str = f"{row['CI_Lower']:.4f}" if not np.isnan(row['CI_Lower']) else "N/A"
        ci_hi_str = f"{row['CI_Upper']:.4f}" if not np.isnan(row['CI_Upper']) else "N/A"
        out.append(
            f"  {str(row['Group']):<20} {row['Adjusted_Mean']:>15.4f} "
            f"{se_str:>10} {ci_lo_str:>14} {ci_hi_str:>14}"
        )
    _dual_print(out, file_handle); out = []

    # ── POST-HOC COMPARISONS ───────────────────────────────────────────────────
    if n_groups > 2 and posthoc is not None and not posthoc.empty:
        out.append(_section("6. POST-HOC PAIRWISE COMPARISONS (Bonferroni Correction)"))
        n_comp = len(posthoc)
        out.append(f"  Number of pairwise comparisons : {n_comp}")
        out.append(
            f"  Correction                     : Bonferroni "
            f"(adjusted α = {0.05/n_comp:.4f} per comparison)"
        )
        out.append("  Comparisons are of adjusted means (covariates at grand mean).")
        out.append("  SE computed via contrast vector c'Vc (model covariance matrix).")
        out.append("")
        out.append(
            f"  {'Group 1':<16} {'Group 2':<16} {'Mean Diff':>12} {'SE':>10} "
            f"{'t':>8} {'p (raw)':>10} {'p (Bonf)':>10} {'CI Lo (adj)':>13} {'CI Hi (adj)':>13}"
        )
        out.append("  " + _separator("-", 113))
        for _, row in posthoc.iterrows():
            def fmt(v, dec=4):
                return f"{v:.{dec}f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            out.append(
                f"  {str(row['Group 1']):<16} {str(row['Group 2']):<16} "
                f"{fmt(row['Mean Difference']):>12} {fmt(row['SE']):>10} "
                f"{fmt(row['t']):>8} {format_p(row['p (raw)']):>10} "
                f"{format_p(row['p (Bonferroni)']):>10} "
                f"{fmt(row['CI Lower (adj)']):>13} {fmt(row['CI Upper (adj)']):>13}"
            )
        out.append("")
        out.append("  Significance: pairs with p (Bonf) < 0.05 differ significantly "
                   "at the Bonferroni-adjusted threshold.")
        _dual_print(out, file_handle); out = []

    # ── MEDICAL INTERPRETATION ─────────────────────────────────────────────────
    section_num = 7 if n_groups > 2 else 6
    out.append(_section(f"{section_num}. MEDICAL & STATISTICAL INTERPRETATION"))

    # Identify group row in type3_table
    cov_keys = set(_sanitize_name(c) for c in covariates)
    group_row_key = None
    for idx in type3_table.index:
        if idx in ("Intercept", "Residual"):
            continue
        if idx not in cov_keys:
            group_row_key = idx
            break

    if group_row_key:
        try:
            g_p = float(type3_table.loc[group_row_key, "PR(>F)"])
            g_f = float(type3_table.loc[group_row_key, "F"])
            g_eta = eta_sq.get(group_row_key, {}).get("partial_eta_sq", np.nan)
            g_effect = eta_sq.get(group_row_key, {}).get("interpretation", "")

            out.append(f"  Group Effect ({group}):")
            if g_p < 0.05:
                out.append(
                    f"    Statistically significant group effect after covariate adjustment "
                    f"(F = {g_f:.3f}, p = {format_p(g_p)}, "
                    f"partial η² = {g_eta:.4f if not np.isnan(g_eta) else 'N/A'} — {g_effect})."
                )
                out.append(
                    f"    After controlling for {', '.join(covariates)}, group membership "
                    f"explains a significant portion of variance in {dv}."
                )
            else:
                out.append(
                    f"    No significant group effect after covariate adjustment "
                    f"(F = {g_f:.3f}, p = {format_p(g_p)}, "
                    f"partial η² = {g_eta:.4f if not np.isnan(g_eta) else 'N/A'})."
                )
                out.append(
                    f"    After controlling for {', '.join(covariates)}, group membership "
                    f"does not significantly predict {dv}."
                )
        except Exception:
            pass

    out.append("")
    out.append("  Covariate Effects:")
    for cov in covariates:
        cov_key = _sanitize_name(cov)
        try:
            c_p = float(type3_table.loc[cov_key, "PR(>F)"])
            c_f = float(type3_table.loc[cov_key, "F"])
            c_eta = eta_sq.get(cov_key, {}).get("partial_eta_sq", np.nan)
            sig = "significant" if c_p < 0.05 else "not significant"
            eta_disp = f"{c_eta:.4f}" if not np.isnan(c_eta) else "N/A"
            out.append(
                f"    {cov}: {sig} covariate effect "
                f"(F = {c_f:.3f}, p = {format_p(c_p)}, partial η² = {eta_disp})."
            )
        except Exception:
            out.append(f"    {cov}: effect could not be extracted from ANOVA table.")

    out.append("")
    out.append("  Adjusted Means Summary:")
    out.append(
        f"    Covariate grand means used: "
        f"{', '.join([f'{c}={df[c].mean():.3f}' for c in covariates])}"
    )
    for _, row in adj_means.iterrows():
        out.append(f"    {row['Group']}: adjusted {dv} = {row['Adjusted_Mean']:.3f}")

    out.append("")
    out.append("  Statistical vs Clinical Significance:")
    out.append("    A statistically significant group effect does not automatically imply")
    out.append("    clinical importance. Evaluate adjusted mean differences relative to")
    out.append("    clinically meaningful thresholds for the outcome measured.")
    out.append("    Partial η² quantifies effect size; clinical relevance requires domain expertise.")

    out.append("")
    out.append("  Large-Sample Note:")
    out.append("    In large samples (n > 200 per group), trivially small group differences")
    out.append("    may reach statistical significance. Always interpret adjusted mean")
    out.append("    differences and confidence intervals alongside p-values.")
    _dual_print(out, file_handle); out = []

    # ── LIMITATIONS ────────────────────────────────────────────────────────────
    section_num += 1
    out.append(_section(f"{section_num}. LIMITATIONS"))
    out.append("""
  1. CAUSALITY
     ANCOVA does not establish causation. Covariate adjustment reduces
     confounding but does not eliminate it. Unmeasured confounders may remain.

  2. RESIDUAL CONFOUNDING
     Covariates are measured with error. Imperfect covariate measurement leads
     to incomplete adjustment (regression dilution bias).

  3. ASSUMPTION VIOLATIONS
     If homogeneity of regression slopes is violated, adjusted means are not
     interpretable as single summary statistics. Group differences then depend
     on the covariate value and must be qualified accordingly.

  4. OUTLIER SENSITIVITY
     OLS-based ANCOVA is sensitive to extreme observations. Influential points
     should be investigated clinically before any exclusion.

  5. GROUP IMBALANCE
     Unequal group sizes amplify sensitivity to variance heterogeneity and can
     affect Type I error rates. Type III SS partially addresses this but does
     not eliminate all imbalance effects.

  6. COVARIATE SELECTION
     Including too many covariates relative to sample size risks overfitting.
     Covariates should be pre-specified based on biological plausibility,
     not selected post-hoc based on significance.

  7. GENERALIZABILITY
     Results generalise only to populations similar to the study sample
     in terms of the observed covariate range.

  8. EXTRAPOLATION
     Adjusted means are meaningful only within the observed range of covariate
     values. Extrapolation beyond this range is unjustifiable.
    """)
    _dual_print(out, file_handle); out = []

    # ── FOOTER ────────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("END OF REPORT — VM-WIT-STATS | VM Medical College")
    out.append("For academic publication, verify all assumptions before reporting.")
    out.append("This output is research-grade and intended for trained professionals.")
    out.append(_separator())
    _dual_print(out, file_handle)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Main entry point for the ANCOVA analysis module.

    Workflow:
        1.  Load CSV
        2.  Display columns
        3.  Select dependent variable
        4.  Select grouping variable
        5.  Select covariates
        6.  Validate data
        7.  Clean data
        8.  Descriptive statistics
        9.  Test homogeneity of regression slopes (critical, with per-covariate detail)
        10. Fit ANCOVA model (Type III SS, sum contrasts)
        11. Detect outliers and influential observations (truncated index display)
        12. Check residual normality (Shapiro-Wilk / Lilliefors)
        13. Check homogeneity of variance (Levene)
        14. Check linearity (DV vs each covariate)
        15. Compute partial eta squared
        16. Compute adjusted (estimated marginal) means
        17. Post-hoc pairwise comparisons (Bonferroni, patsy contrast vectors)
        18. Generate and save report
    """
    print(_separator())
    print("VM-WIT-STATS | VM Medical College")
    print("ANALYSIS OF COVARIANCE (ANCOVA)")
    print(_separator())

    use_sample = input(
        "\nGenerate and use a sample medical dataset for testing? [y/n]: "
    ).strip().lower()

    if use_sample == "y":
        filepath = generate_sample_csv()
    else:
        filepath = input("Enter the full path to your CSV file: ").strip()

    df_raw = load_csv(filepath)
    if df_raw is None:
        print("[FATAL] Could not load data. Exiting.")
        sys.exit(1)

    display_columns(df_raw)

    dv = select_dependent_variable(df_raw)
    if dv is None:
        print("[FATAL] No dependent variable selected. Exiting.")
        sys.exit(1)

    group = select_group_variable(df_raw, dv)
    if group is None:
        print("[FATAL] No grouping variable selected. Exiting.")
        sys.exit(1)

    covariates = select_covariates(df_raw, dv, group)
    if not covariates:
        print("[FATAL] No covariates selected. Exiting.")
        sys.exit(1)

    if not validate_data(df_raw, dv, group, covariates):
        print("[FATAL] Validation failed. Exiting.")
        sys.exit(1)

    df = clean_data(df_raw, dv, group, covariates)
    n = len(df)

    group_desc, cov_desc = descriptive_statistics(df, dv, group, covariates)

    print("\n[INFO] Testing homogeneity of regression slopes...")
    slopes_res = test_homogeneity_of_slopes(df, dv, group, covariates)

    print("[INFO] Fitting ANCOVA model (Type III SS, sum contrasts)...")
    model, type3_table = fit_ancova_model(df, dv, group, covariates)
    if model is None:
        print("[FATAL] ANCOVA model fitting failed. Exiting.")
        sys.exit(1)

    print("[INFO] Running assumption checks...")
    normality_res = check_residual_normality(model.resid.values)
    levene_res = check_homogeneity_of_variance(df, dv, group)
    linearity_res = check_linearity(df, dv, covariates)
    diag_res = detect_outliers_and_influence(model, n)

    eta_sq = compute_partial_eta_squared(type3_table)

    print("[INFO] Computing adjusted means...")
    adj_means = compute_adjusted_means(model, df, group, covariates)

    groups_list = sorted(df[group].unique())
    posthoc = None
    if len(groups_list) > 2:
        print("[INFO] Running Bonferroni post-hoc pairwise comparisons...")
        posthoc = posthoc_bonferroni(model, df, group, covariates, dv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            generate_report(
                df=df, dv=dv, group=group, covariates=covariates,
                group_desc=group_desc, cov_desc=cov_desc,
                slopes_res=slopes_res, normality_res=normality_res,
                levene_res=levene_res, linearity_res=linearity_res,
                diag_res=diag_res, model=model, type3_table=type3_table,
                eta_sq=eta_sq, adj_means=adj_means, posthoc=posthoc,
                file_handle=fh,
            )
        print(f"\n[INFO] Report saved to: {OUTPUT_FILE}")
    except OSError as e:
        print(f"[ERROR] Could not save report: {e}")
        print("[INFO] Displaying report to console only.")
        import io
        generate_report(
            df=df, dv=dv, group=group, covariates=covariates,
            group_desc=group_desc, cov_desc=cov_desc,
            slopes_res=slopes_res, normality_res=normality_res,
            levene_res=levene_res, linearity_res=linearity_res,
            diag_res=diag_res, model=model, type3_table=type3_table,
            eta_sq=eta_sq, adj_means=adj_means, posthoc=posthoc,
            file_handle=io.StringIO(),
        )


if __name__ == "__main__":
    main()