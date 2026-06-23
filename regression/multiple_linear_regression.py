"""
multiple_linear_regression.py
=============================
VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit

Multiple Linear Regression Module
Model: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ

Intended for medical researchers, clinical researchers, and academic publications.
Production-grade | Research-grade | Publication-grade | Biostatistics-grade

Author: VM-WIT-STATS Toolkit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, lilliefors as sm_lilliefors
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "multiple_linear_regression_result.txt")


# ==============================================================================
# SAMPLE DATA GENERATOR
# ==============================================================================

def generate_sample_csv(filepath: str = "sample_medical_data.csv") -> str:
    """
    Generate a realistic synthetic medical dataset for testing.

    Variables:
        Age          - Patient age in years
        BMI          - Body Mass Index
        Creatinine   - Serum creatinine (mg/dL)
        BloodPressure - Systolic blood pressure (mmHg)

    Returns:
        str: Path to saved CSV file.
    """
    np.random.seed(42)
    n = 120

    age = np.random.normal(55, 12, n).clip(20, 90)
    bmi = np.random.normal(27, 5, n).clip(16, 50)
    creatinine = np.random.exponential(1.0, n).clip(0.4, 8.0)

    noise = np.random.normal(0, 8, n)
    blood_pressure = (
        90
        + 0.45 * age
        + 0.60 * bmi
        + 3.50 * creatinine
        + noise
    ).clip(80, 200)

    df = pd.DataFrame({
        "Age": np.round(age, 1),
        "BMI": np.round(bmi, 1),
        "Creatinine": np.round(creatinine, 2),
        "BloodPressure": np.round(blood_pressure, 1),
    })

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
    """Print to console and write to file simultaneously."""
    for line in lines:
        print(line)
        file_handle.write(line + "\n")


# ==============================================================================
# FILE LOADING
# ==============================================================================

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame with comprehensive validation.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset, or None on failure.
    """
    if not filepath or not isinstance(filepath, str):
        print("[ERROR] Invalid file path provided.")
        return None

    filepath = filepath.strip()

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    if not filepath.lower().endswith(".csv"):
        print("[WARNING] File does not have a .csv extension. Attempting to load anyway.")

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
# COLUMN SELECTION
# ==============================================================================

def display_columns(df: pd.DataFrame) -> None:
    """Display available columns with their data types."""
    print("\n" + _separator("-"))
    print("AVAILABLE COLUMNS")
    print(_separator("-"))
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        print(f"  [{i:>2}] {col:<30} dtype={dtype}  missing={n_missing}")
    print(_separator("-"))


def select_outcome(df: pd.DataFrame) -> str:
    """
    Interactively select the outcome (dependent) variable.

    Parameters:
        df (pd.DataFrame): Loaded dataset.

    Returns:
        str: Column name of selected outcome variable, or None on failure.
    """
    cols = list(df.columns)
    while True:
        try:
            raw = input("\nEnter the column name or number for the OUTCOME variable (Y): ").strip()
            if not raw:
                print("[ERROR] No input provided.")
                continue

            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(cols):
                    outcome = cols[idx]
                else:
                    print(f"[ERROR] Number out of range. Enter 1–{len(cols)}.")
                    continue
            elif raw in cols:
                outcome = raw
            else:
                print(f"[ERROR] Column '{raw}' not found.")
                continue

            if not pd.api.types.is_numeric_dtype(df[outcome]):
                print(f"[ERROR] '{outcome}' is not numeric. Select a continuous numeric column.")
                continue

            print(f"[INFO] Outcome selected: {outcome}")
            return outcome

        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            continue


def select_predictors(df: pd.DataFrame, outcome: str) -> list:
    """
    Interactively select two or more predictor (independent) variables.

    Parameters:
        df (pd.DataFrame): Loaded dataset.
        outcome (str): Name of the outcome variable (excluded from selection).

    Returns:
        list: List of selected predictor column names, or None on failure.
    """
    cols = [c for c in df.columns if c != outcome]
    print(f"\nAvailable predictors (excluding outcome '{outcome}'):")
    for i, col in enumerate(cols, 1):
        dtype = df[col].dtype
        print(f"  [{i:>2}] {col:<30} dtype={dtype}")

    while True:
        try:
            raw = input(
                "\nEnter predictor column names or numbers separated by commas\n"
                "(minimum 2 required): "
            ).strip()

            if not raw:
                print("[ERROR] No input provided.")
                continue

            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            if len(tokens) < 2:
                print("[ERROR] Please select at least 2 predictor variables.")
                continue

            predictors = []
            valid = True
            for token in tokens:
                if token.isdigit():
                    idx = int(token) - 1
                    if 0 <= idx < len(cols):
                        predictors.append(cols[idx])
                    else:
                        print(f"[ERROR] Number {token} out of range. Valid range: 1–{len(cols)}.")
                        valid = False
                        break
                elif token in cols:
                    predictors.append(token)
                else:
                    print(f"[ERROR] Column '{token}' not found or is the outcome variable.")
                    valid = False
                    break

            if not valid:
                continue

            # Check duplicates
            if len(predictors) != len(set(predictors)):
                print("[ERROR] Duplicate predictors selected. Each predictor must be unique.")
                continue

            # Validate numeric
            non_numeric = [p for p in predictors if not pd.api.types.is_numeric_dtype(df[p])]
            if non_numeric:
                print(f"[ERROR] Non-numeric predictors selected: {non_numeric}. All predictors must be numeric.")
                continue

            print(f"[INFO] Predictors selected: {predictors}")
            return predictors

        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Selection cancelled.")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            continue


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_variables(df: pd.DataFrame, outcome: str, predictors: list) -> bool:
    """
    Comprehensive validation of selected variables.

    Checks:
        - Columns exist
        - Numeric types
        - Constant columns
        - Sufficient sample size
        - Missing and infinite values (with warnings)

    Parameters:
        df (pd.DataFrame): Full dataset.
        outcome (str): Outcome column name.
        predictors (list): List of predictor column names.

    Returns:
        bool: True if validation passes (possibly with warnings), False on fatal errors.
    """
    all_vars = [outcome] + predictors
    fatal = False

    # Column existence
    for var in all_vars:
        if var not in df.columns:
            print(f"[ERROR] Column '{var}' not found in dataset.")
            fatal = True

    if fatal:
        return False

    # Numeric check
    for var in all_vars:
        if not pd.api.types.is_numeric_dtype(df[var]):
            print(f"[ERROR] Variable '{var}' is not numeric.")
            fatal = True

    if fatal:
        return False

    # Infinite values
    for var in all_vars:
        n_inf = np.isinf(df[var].replace([None], np.nan).dropna()).sum()
        if n_inf > 0:
            print(f"[WARNING] Variable '{var}' contains {n_inf} infinite value(s). These will be removed.")

    # Missing values
    for var in all_vars:
        n_miss = df[var].isna().sum()
        if n_miss > 0:
            pct = 100 * n_miss / len(df)
            print(f"[WARNING] Variable '{var}' has {n_miss} missing values ({pct:.1f}%). These will be excluded.")

    # Work on clean subset
    subset = df[all_vars].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(subset)

    if n < 3:
        print(f"[ERROR] Insufficient observations after cleaning: n = {n}. Minimum required: 3.")
        return False
    elif n < 10:
        print(f"[WARNING SEVERE] Very small sample size: n = {n}. Results are highly unreliable.")
    elif n < 20:
        print(f"[WARNING] Small sample size: n = {n}. Interpret results with caution.")
    elif n < 30:
        print(f"[CAUTION] Sample size: n = {n}. Results may be unstable.")

    k = len(predictors)
    # Rule of thumb: at least 10–20 observations per predictor
    if n < 10 * k:
        print(f"[WARNING] Sample size (n={n}) is less than 10× the number of predictors (k={k}). "
              f"Risk of overfitting. Recommend n ≥ {10 * k}.")
    if k >= n - 1:
        print(f"[ERROR] Number of predictors (k={k}) equals or exceeds available degrees of freedom (n-1={n-1}). "
              f"Model cannot be estimated.")
        return False

    # Constant columns
    for var in all_vars:
        clean_vals = subset[var]
        if clean_vals.nunique() <= 1:
            print(f"[ERROR] Variable '{var}' is constant (no variance). Cannot be used in regression.")
            fatal = True

    if fatal:
        return False

    print(f"[INFO] Validation passed. Usable observations: {n}")
    return True


# ==============================================================================
# DATA CLEANING
# ==============================================================================

def clean_data(df: pd.DataFrame, outcome: str, predictors: list) -> pd.DataFrame:
    """
    Extract relevant variables, remove infinities and missing values.

    Parameters:
        df (pd.DataFrame): Full dataset.
        outcome (str): Outcome column name.
        predictors (list): List of predictor column names.

    Returns:
        pd.DataFrame: Cleaned dataset with only selected variables.
    """
    all_vars = [outcome] + predictors
    subset = df[all_vars].copy()
    n_before = len(subset)
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    n_after = len(subset)
    removed = n_before - n_after
    if removed > 0:
        print(f"[INFO] Removed {removed} row(s) due to missing or infinite values. "
              f"Remaining observations: {n_after}")
    subset = subset.reset_index(drop=True)
    return subset


# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

def descriptive_statistics(df: pd.DataFrame, outcome: str, predictors: list) -> pd.DataFrame:
    """
    Compute descriptive statistics for the outcome and all predictors.

    Statistics:
        N, Mean, Median, SD, SE, Min, Max, Q1, Q3, IQR, Skewness, Kurtosis

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        outcome (str): Outcome column name.
        predictors (list): List of predictor column names.

    Returns:
        pd.DataFrame: Table of descriptive statistics.
    """
    all_vars = [outcome] + predictors
    rows = []
    for var in all_vars:
        s = df[var]
        n = len(s)
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        rows.append({
            "Variable": var,
            "Role": "Outcome" if var == outcome else "Predictor",
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
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s)),  # excess kurtosis
        })
    return pd.DataFrame(rows)


# ==============================================================================
# MODEL FITTING
# ==============================================================================

def fit_ols_model(df: pd.DataFrame, outcome: str, predictors: list):
    """
    Fit Ordinary Least Squares multiple linear regression using statsmodels.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        outcome (str): Outcome column name.
        predictors (list): List of predictor column names.

    Returns:
        tuple: (model_result, X_with_const, y)
            - model_result: statsmodels OLS fitted result
            - X_with_const: Design matrix including intercept column
            - y: Outcome Series
    """
    y = df[outcome]
    X = df[predictors]
    X_const = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, X_const)
        result = model.fit()
        return result, X_const, y
    except Exception as e:
        print(f"[ERROR] Model fitting failed: {e}")
        return None, None, None


def fit_robust_model(result, X_const, y):
    """
    Refit OLS with HC3 heteroscedasticity-consistent (robust) standard errors.

    HC3 is the preferred correction for small-to-moderate sample sizes
    (MacKinnon & White, 1985). It does not alter coefficient estimates,
    only their standard errors and derived inference.

    Parameters:
        result: Original OLS result.
        X_const: Design matrix.
        y: Outcome Series.

    Returns:
        Robust OLS result object.
    """
    try:
        robust_result = result.get_robustcov_results(cov_type="HC3")
        return robust_result
    except Exception as e:
        print(f"[WARNING] Robust SE computation failed: {e}")
        return None


# ==============================================================================
# MULTICOLLINEARITY — VIF
# ==============================================================================

def compute_vif(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors (VIF) for all predictors.

    VIF = 1 / (1 - R²ⱼ) where R²ⱼ is the R² from regressing predictor j
    on all other predictors.

    Interpretation:
        VIF < 5      : Acceptable
        5 ≤ VIF < 10 : Moderate concern
        VIF ≥ 10     : Serious multicollinearity

    Parameters:
        df (pd.DataFrame): Cleaned dataset with predictors only.
        predictors (list): List of predictor column names.

    Returns:
        pd.DataFrame: VIF table with interpretation.
    """
    X = df[predictors].copy()
    X_const = sm.add_constant(X, has_constant="add")

    vif_data = []
    for i, col in enumerate(predictors):
        col_idx = list(X_const.columns).index(col)
        try:
            vif_val = variance_inflation_factor(X_const.values, col_idx)
        except Exception:
            vif_val = np.nan

        if np.isnan(vif_val) or np.isinf(vif_val):
            interpretation = "UNDEFINED (perfect collinearity suspected)"
        elif vif_val < 5:
            interpretation = "Acceptable"
        elif vif_val < 10:
            interpretation = "Moderate concern — review model"
        else:
            interpretation = "SERIOUS — consider removing this predictor"

        vif_data.append({
            "Predictor": col,
            "VIF": vif_val,
            "Interpretation": interpretation,
        })

    return pd.DataFrame(vif_data)


def compute_predictor_correlation_matrix(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix among all predictor variables.

    Used alongside VIF to identify which specific predictor pairs drive
    multicollinearity. VIF quantifies the severity of multicollinearity
    for each predictor; the correlation matrix identifies which pairs
    of predictors are the source.

    Note: VIF alone cannot identify which predictor pairs are correlated.
    The correlation matrix provides that structural detail.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        predictors (list): List of predictor column names.

    Returns:
        pd.DataFrame: Symmetric Pearson correlation matrix, predictors × predictors.
    """
    return df[predictors].corr(method="pearson")


# ==============================================================================
# ASSUMPTION DIAGNOSTICS
# ==============================================================================

def check_linearity_reset(result) -> dict:
    """
    Ramsey RESET Test for functional form misspecification (linearity).

    H₀: Model functional form is correct (linearity holds).
    H₁: Functional form is misspecified (non-linearity present).

    Uses power=3 to augment the model with both Ŷ² and Ŷ³, providing
    sensitivity to both quadratic and cubic departures from linearity.
    This is preferred over power=2 (quadratic only) for publication-grade
    reporting where non-monotone dose–response relationships are plausible
    (Ramsey, 1969; Greene, Econometric Analysis, 7th ed.).

    Parameters:
        result: OLS fitted result.

    Returns:
        dict with F-statistic, p-value, and interpretation.
    """
    try:
        reset_result = linear_reset(result, power=3, use_f=True)
        f_stat = float(reset_result.fvalue)
        p_val = float(reset_result.pvalue)
        if p_val > 0.05:
            interp = (
                "No significant evidence of linearity violation (p > 0.05). "
                "Linearity assumption is supported (tested against quadratic and cubic departures)."
            )
        else:
            interp = (
                "Significant linearity violation detected (p ≤ 0.05). "
                "The model may be misspecified for quadratic or cubic relationships. "
                "Consider non-linear transformations, interaction terms, or polynomial terms."
            )
        return {"F_statistic": f_stat, "p_value": p_val, "interpretation": interp}
    except Exception as e:
        return {
            "F_statistic": np.nan,
            "p_value": np.nan,
            "interpretation": f"RESET test could not be computed: {e}",
        }


def check_residual_normality(residuals: np.ndarray) -> dict:
    """
    Test residual normality with sample-size-appropriate method.

    For n ≤ 2000 : Shapiro-Wilk test (most powerful for moderate samples).
    For n > 2000 : Lilliefors test (statsmodels implementation).

    Why NOT scipy.stats.kstest() for large samples:
        The classical Kolmogorov-Smirnov test assumes distribution parameters
        (mean, SD) are known a priori. For regression residuals, parameters are
        always estimated from the same sample being tested. Plugging estimated
        parameters into kstest() produces conservative p-values (the null
        distribution is incorrect), systematically under-rejecting the null.
        The Lilliefors correction provides the appropriate null distribution
        for this case (Lilliefors, 1967).

    Large-sample caveat:
        For n > 5000, normality tests have near-certain power to detect
        trivially small deviations that do not affect OLS inference in practice.
        OLS coefficient estimates are protected by the Central Limit Theorem
        at large sample sizes regardless of residual normality.

    Parameters:
        residuals (np.ndarray): OLS residuals.

    Returns:
        dict with test name, statistic, p-value, and interpretation.
    """
    try:
        n = len(residuals)

        if n <= 2000:
            stat, p_val = stats.shapiro(residuals)
            test_name = "Shapiro-Wilk"
        else:
            # Lilliefors is the correct KS-based test when parameters are
            # estimated from the data. pvalmethod="approx" uses the
            # asymptotic approximation, valid for n > 2000.
            stat, p_val = sm_lilliefors(residuals, dist="norm", pvalmethod="approx")
            test_name = "Lilliefors (KS with estimated parameters)"

        large_n_caveat = ""
        if n > 5000:
            large_n_caveat = (
                " NOTE: With n > 5000, normality tests detect trivially small deviations. "
                "OLS inference is protected by the CLT at this sample size; "
                "a significant result here does not necessarily indicate a practical problem."
            )

        if p_val > 0.05:
            interp = (
                f"{test_name}: Residuals are consistent with normality (p > 0.05). "
                f"Normality assumption is supported.{large_n_caveat}"
            )
        else:
            interp = (
                f"{test_name}: Residuals significantly deviate from normality (p ≤ 0.05). "
                "OLS coefficient estimates remain unbiased (CLT), but inference on small samples "
                "may be affected. Consider robust standard errors or bootstrap confidence intervals."
                f"{large_n_caveat}"
            )

        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_val),
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "test": "Unknown",
            "statistic": np.nan,
            "p_value": np.nan,
            "interpretation": f"Normality test failed: {e}",
        }


def check_homoscedasticity(result, X_const, y) -> dict:
    """
    Breusch-Pagan Test for heteroscedasticity.

    H₀: Residuals are homoscedastic (constant variance).
    H₁: Residuals are heteroscedastic (non-constant variance).

    Parameters:
        result: OLS fitted result.
        X_const: Design matrix with intercept.
        y: Outcome Series.

    Returns:
        dict with LM statistic, p-value, heteroscedasticity flag, and interpretation.
    """
    try:
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(result.resid, X_const)
        heteroscedastic = lm_p <= 0.05
        if not heteroscedastic:
            interp = (
                "Breusch-Pagan: No significant heteroscedasticity detected (p > 0.05). "
                "Homoscedasticity assumption is supported."
            )
        else:
            interp = (
                "Breusch-Pagan: Significant heteroscedasticity detected (p ≤ 0.05). "
                "Standard OLS standard errors are invalid. HC3 robust standard errors "
                "will be computed and reported."
            )
        return {
            "LM_statistic": float(lm_stat),
            "LM_p_value": float(lm_p),
            "F_statistic": float(f_stat),
            "F_p_value": float(f_p),
            "heteroscedastic": heteroscedastic,
            "interpretation": interp,
        }
    except Exception as e:
        return {
            "LM_statistic": np.nan,
            "LM_p_value": np.nan,
            "F_statistic": np.nan,
            "F_p_value": np.nan,
            "heteroscedastic": False,
            "interpretation": f"Breusch-Pagan test failed: {e}",
        }


def check_independence_dw(residuals: np.ndarray) -> dict:
    """
    Durbin-Watson Statistic for autocorrelation of residuals.

    DW ≈ 2   : No autocorrelation
    DW < 1.5 : Positive autocorrelation
    DW > 2.5 : Negative autocorrelation

    Important caveat: DW is designed for time-series data.
    For cross-sectional medical data, autocorrelation is unlikely
    but clustering effects may exist.

    Parameters:
        residuals (np.ndarray): OLS residuals.

    Returns:
        dict with DW statistic and interpretation.
    """
    try:
        dw = float(durbin_watson(residuals))
        if dw < 1.5:
            interp = (
                f"DW = {dw:.3f}: Possible positive autocorrelation. "
                "For cross-sectional data, this may indicate data entry ordering artifacts "
                "or unaccounted clustering."
            )
        elif dw > 2.5:
            interp = (
                f"DW = {dw:.3f}: Possible negative autocorrelation. "
                "For cross-sectional data, verify row ordering is not systematic."
            )
        else:
            interp = (
                f"DW = {dw:.3f}: No evidence of autocorrelation (1.5 < DW < 2.5). "
                "Independence of errors assumption is supported."
            )
        caveat = (
            "NOTE: Durbin-Watson is primarily designed for time-series data. "
            "For cross-sectional clinical datasets, low DW may reflect row ordering "
            "rather than true serial correlation."
        )
        return {"DW_statistic": dw, "interpretation": interp, "caveat": caveat}
    except Exception as e:
        return {
            "DW_statistic": np.nan,
            "interpretation": f"Durbin-Watson computation failed: {e}",
            "caveat": "",
        }


def detect_outliers_leverage_influence(result, X_const, n: int, k: int) -> dict:
    """
    Detect outliers, high-leverage, and influential observations.

    Methods:
        Outliers (moderate) : 2 < |Studentized Residual| ≤ 3
        Outliers (strong)   : |Studentized Residual| > 3
        Leverage            : Hat values > 2(k+1)/n
        Influence           : Cook's Distance > 4/n

    Two-level studentized residual flagging follows Belsley, Kuh & Welsch (1980)
    and Cook & Weisberg (1982): observations with 2 < |SR| ≤ 3 represent a
    moderate concern zone warranting inspection; |SR| > 3 represents a strong
    outlier requiring clinical verification.

    Parameters:
        result: OLS fitted result.
        X_const: Design matrix.
        n (int): Sample size.
        k (int): Number of predictors (excluding intercept).

    Returns:
        dict with arrays and flagged indices for both outlier tiers.
    """
    try:
        influence = OLSInfluence(result)
        student_resid = influence.resid_studentized_external
        hat_values = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        leverage_cutoff = 2 * (k + 1) / n
        cooks_cutoff = 4 / n

        # Two-tier outlier detection
        strong_outlier_idx = np.where(np.abs(student_resid) > 3)[0]
        moderate_outlier_idx = np.where(
            (np.abs(student_resid) > 2) & (np.abs(student_resid) <= 3)
        )[0]
        high_lev_idx = np.where(hat_values > leverage_cutoff)[0]
        influential_idx = np.where(cooks_d > cooks_cutoff)[0]

        return {
            "studentized_residuals": student_resid,
            "hat_values": hat_values,
            "cooks_distance": cooks_d,
            "outlier_indices": strong_outlier_idx,
            "moderate_outlier_indices": moderate_outlier_idx,
            "high_leverage_indices": high_lev_idx,
            "influential_indices": influential_idx,
            "leverage_cutoff": leverage_cutoff,
            "cooks_cutoff": cooks_cutoff,
            "n_outliers": len(strong_outlier_idx),
            "n_moderate_outliers": len(moderate_outlier_idx),
            "n_high_leverage": len(high_lev_idx),
            "n_influential": len(influential_idx),
        }
    except Exception as e:
        print(f"[WARNING] Outlier/leverage/influence diagnostics failed: {e}")
        return {
            "studentized_residuals": np.array([]),
            "hat_values": np.array([]),
            "cooks_distance": np.array([]),
            "outlier_indices": np.array([]),
            "moderate_outlier_indices": np.array([]),
            "high_leverage_indices": np.array([]),
            "influential_indices": np.array([]),
            "leverage_cutoff": np.nan,
            "cooks_cutoff": np.nan,
            "n_outliers": 0,
            "n_moderate_outliers": 0,
            "n_high_leverage": 0,
            "n_influential": 0,
        }


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def format_p_value(p: float) -> str:
    """Format p-value for reporting, using < 0.001 threshold."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def generate_report(
    df: pd.DataFrame,
    outcome: str,
    predictors: list,
    desc_stats: pd.DataFrame,
    result,
    robust_result,
    vif_df: pd.DataFrame,
    reset_res: dict,
    normality_res: dict,
    homosced_res: dict,
    dw_res: dict,
    diag_res: dict,
    file_handle,
) -> None:
    """
    Generate the complete analysis report to console and file simultaneously.

    Parameters:
        df: Cleaned dataset.
        outcome: Outcome variable name.
        predictors: List of predictor names.
        desc_stats: Descriptive statistics DataFrame.
        result: OLS fitted result.
        robust_result: HC3 robust fitted result (or None).
        vif_df: VIF table.
        reset_res: RESET test results.
        normality_res: Residual normality results.
        homosced_res: Breusch-Pagan results.
        dw_res: Durbin-Watson results.
        diag_res: Outlier/leverage/influence results.
        file_handle: Open file handle for output.
    """
    n = len(df)
    k = len(predictors)
    out = []

    # ── HEADER ──────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("VM-WIT-STATS | VM Medical College — Medical Statistics Toolkit")
    out.append("MULTIPLE LINEAR REGRESSION ANALYSIS")
    out.append(_separator())
    out.append(f"Outcome Variable  : {outcome}")
    out.append(f"Predictor Variables ({k}): {', '.join(predictors)}")
    out.append(f"Sample Size (n)   : {n}")
    out.append(f"Predictors (k)    : {k}")
    out.append(f"Model             : Y = β₀ + β₁X₁ + β₂X₂ + ... + β{k}X{k}")
    _dual_print(out, file_handle); out = []

    # ── DESCRIPTIVE STATISTICS ───────────────────────────────────────────────
    out.append(_section("1. DESCRIPTIVE STATISTICS"))
    header = (
        f"{'Variable':<22} {'Role':<10} {'N':>5} {'Mean':>10} {'Median':>10} "
        f"{'SD':>10} {'SE':>10} {'Min':>10} {'Max':>10} "
        f"{'Q1':>10} {'Q3':>10} {'IQR':>10} {'Skew':>7} {'Kurt':>7}"
    )
    out.append(header)
    out.append(_separator("-"))
    for _, row in desc_stats.iterrows():
        out.append(
            f"{row['Variable']:<22} {row['Role']:<10} {int(row['N']):>5} "
            f"{row['Mean']:>10.3f} {row['Median']:>10.3f} {row['SD']:>10.3f} "
            f"{row['SE']:>10.3f} {row['Min']:>10.3f} {row['Max']:>10.3f} "
            f"{row['Q1']:>10.3f} {row['Q3']:>10.3f} {row['IQR']:>10.3f} "
            f"{row['Skewness']:>7.3f} {row['Kurtosis']:>7.3f}"
        )
    _dual_print(out, file_handle); out = []

    # ── MODEL PERFORMANCE ────────────────────────────────────────────────────
    out.append(_section("2. MODEL PERFORMANCE"))

    r2 = result.rsquared
    r2_adj = result.rsquared_adj
    f_stat = result.fvalue
    f_p = result.f_pvalue
    aic = result.aic
    bic = result.bic
    rse = np.sqrt(result.mse_resid)

    out.append(f"  R²                      : {r2:.4f}  ({100*r2:.2f}% of variance in {outcome} explained)")
    out.append(f"  Adjusted R²             : {r2_adj:.4f}  (penalises for number of predictors)")
    out.append(f"  F-statistic             : {f_stat:.4f}")
    out.append(f"  Model p-value           : {format_p_value(f_p)}")
    out.append(f"  Residual Standard Error : {rse:.4f}  (average prediction error in units of {outcome})")
    out.append(f"  AIC                     : {aic:.4f}")
    out.append(f"  BIC                     : {bic:.4f}")
    out.append("")

    out.append("  Interpretation:")
    if f_p < 0.05:
        out.append(
            f"  - The overall regression model is statistically significant "
            f"(F = {f_stat:.3f}, p {format_p_value(f_p)}), indicating that the set of "
            f"predictors collectively explains a significant portion of variance in {outcome}."
        )
    else:
        out.append(
            f"  - The overall regression model is NOT statistically significant "
            f"(F = {f_stat:.3f}, p = {format_p_value(f_p)}). "
            "The predictors do not collectively explain significant variance."
        )

    if r2 >= 0.70:
        r2_comment = "Strong model fit."
    elif r2 >= 0.40:
        r2_comment = "Moderate model fit."
    elif r2 >= 0.15:
        r2_comment = "Weak-to-moderate model fit."
    else:
        r2_comment = "Poor model fit — most variance in the outcome is unexplained."
    out.append(f"  - R² = {r2:.4f}: {r2_comment}")

    diff_r2 = r2 - r2_adj
    if diff_r2 > 0.05:
        out.append(
            f"  - R² vs Adjusted R² discrepancy = {diff_r2:.4f}. "
            "This suggests possible overfitting. Consider model parsimony."
        )
    else:
        out.append(f"  - R² vs Adjusted R² discrepancy = {diff_r2:.4f} (minimal overfitting penalty).")

    _dual_print(out, file_handle); out = []

    # ── OLS COEFFICIENT TABLE ────────────────────────────────────────────────
    out.append(_section("3. REGRESSION COEFFICIENTS (OLS)"))
    out.append(
        f"  {'Variable':<22} {'β (Coef)':>12} {'SE':>10} {'t':>9} "
        f"{'p-value':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}"
    )
    out.append("  " + _separator("-", 93))

    coef_names = result.model.exog_names
    coefs = result.params
    ses = result.bse
    t_vals = result.tvalues
    p_vals = result.pvalues
    ci = result.conf_int()

    for name in coef_names:
        label = "Intercept" if name == "const" else name
        out.append(
            f"  {label:<22} {coefs[name]:>12.4f} {ses[name]:>10.4f} "
            f"{t_vals[name]:>9.4f} {format_p_value(p_vals[name]):>10} "
            f"{ci.loc[name, 0]:>14.4f} {ci.loc[name, 1]:>14.4f}"
        )
    _dual_print(out, file_handle); out = []

    # Individual predictor interpretation
    out.append("\n  Coefficient Interpretation:")
    for pred in predictors:
        if pred not in coefs:
            continue
        b = coefs[pred]
        p = p_vals[pred]
        ci_l = ci.loc[pred, 0]
        ci_u = ci.loc[pred, 1]
        direction = "increases" if b > 0 else "decreases"
        sig = "statistically significant" if p < 0.05 else "NOT statistically significant"

        out.append(f"\n  {pred}:")
        out.append(
            f"    β = {b:.4f}: Holding all other predictors constant, a one-unit increase in "
            f"{pred} is associated with a {abs(b):.4f}-unit {direction} in {outcome}."
        )
        out.append(f"    95% CI: [{ci_l:.4f}, {ci_u:.4f}]")
        out.append(f"    p = {format_p_value(p)}: {sig}.")
    _dual_print(out, file_handle); out = []

    # ── MULTICOLLINEARITY — VIF + CORRELATION MATRIX ─────────────────────────
    out.append(_section("4. MULTICOLLINEARITY DIAGNOSTICS"))
    out.append("  Two complementary diagnostics are reported:")
    out.append("    Predictor Correlation Matrix → identifies which predictor pairs are related")
    out.append("    Variance Inflation Factors   → quantifies multicollinearity severity per predictor")
    out.append("")

    # Predictor Correlation Matrix
    out.append("  [A] PREDICTOR CORRELATION MATRIX (Pearson r)")
    out.append("      Values near ±1.0 between predictors indicate potential collinearity sources.")
    out.append("      VIF alone cannot identify which predictor pairs drive multicollinearity.")
    out.append("")

    corr_mat = compute_predictor_correlation_matrix(df, predictors)
    col_width = 11
    header_row = "      " + " " * 22
    for pred in predictors:
        header_row += f"{pred[:col_width]:>{col_width}}"
    out.append(header_row)
    for pred_row in predictors:
        row_str = f"      {pred_row:<22}"
        for pred_col in predictors:
            val = corr_mat.loc[pred_row, pred_col]
            row_str += f"{val:>{col_width}.3f}"
        out.append(row_str)

    out.append("")
    flagged_pairs = []
    for i, p1 in enumerate(predictors):
        for j, p2 in enumerate(predictors):
            if j <= i:
                continue
            r = corr_mat.loc[p1, p2]
            if abs(r) >= 0.80:
                flagged_pairs.append((p1, p2, r))

    if flagged_pairs:
        out.append("      ⚠ High pairwise correlations (|r| ≥ 0.80) — likely collinearity sources:")
        for p1, p2, r in flagged_pairs:
            out.append(f"        {p1} ↔ {p2}: r = {r:.3f}")
    else:
        out.append("      ✓ No predictor pairs with |r| ≥ 0.80 detected.")

    out.append("")

    # VIF Table
    out.append("  [B] VARIANCE INFLATION FACTORS (VIF)")
    out.append("      VIF = 1/(1 − R²ⱼ), where R²ⱼ = R² from regressing predictor j on all others.")
    out.append("")
    out.append(f"      {'Predictor':<25} {'VIF':>10} {'Interpretation'}")
    out.append("      " + _separator("-", 64))

    serious_mc = []
    moderate_mc = []
    for _, row in vif_df.iterrows():
        vif_str = f"{row['VIF']:.3f}" if not np.isnan(row['VIF']) else "N/A"
        out.append(f"      {row['Predictor']:<25} {vif_str:>10}  {row['Interpretation']}")
        if "SERIOUS" in row["Interpretation"]:
            serious_mc.append(row["Predictor"])
        elif "Moderate" in row["Interpretation"]:
            moderate_mc.append(row["Predictor"])

    out.append("")
    out.append("      Reference:")
    out.append("        VIF < 5      : Acceptable multicollinearity")
    out.append("        5 ≤ VIF < 10 : Moderate concern — review predictors")
    out.append("        VIF ≥ 10     : Serious multicollinearity — coefficient estimates unreliable")
    out.append("")

    if serious_mc:
        out.append(f"      ⚠ SERIOUS MULTICOLLINEARITY: {', '.join(serious_mc)}")
        out.append("        Recommendations:")
        out.append("        - Remove one of the highly correlated predictors.")
        out.append("        - Apply ridge regression or LASSO.")
        out.append("        - Combine correlated predictors into a composite score.")
        out.append("        - Interpret individual coefficients with extreme caution.")
    elif moderate_mc:
        out.append(f"      MODERATE MULTICOLLINEARITY: {', '.join(moderate_mc)}")
        out.append("        Monitor coefficient stability if predictors are added or removed.")
    else:
        out.append("      ✓ No problematic multicollinearity detected.")
    _dual_print(out, file_handle); out = []

    # ── ROBUST STANDARD ERRORS ───────────────────────────────────────────────
    if homosced_res["heteroscedastic"] and robust_result is not None:
        out.append(_section("5. HC3 ROBUST STANDARD ERRORS (Heteroscedasticity Detected)"))
        out.append("  Heteroscedasticity was detected (Breusch-Pagan p ≤ 0.05).")
        out.append("  HC3 robust standard errors are reported below.")
        out.append("  HC3 is the preferred correction for small-to-moderate sample sizes")
        out.append("  (MacKinnon & White, 1985).")
        out.append("  Robust SEs do not change coefficient estimates, only their standard errors.")
        out.append("  Where robust p-values differ from OLS p-values, the robust results")
        out.append("  should be preferred for inference.")
        out.append("")
        out.append(
            f"  {'Variable':<22} {'β (Coef)':>12} {'Robust SE':>12} {'Robust t':>10} "
            f"{'Robust p':>12} {'Robust CI Lo':>14} {'Robust CI Hi':>14}"
        )
        out.append("  " + _separator("-", 98))

        try:
            # Use positional indexing throughout for cross-version compatibility.
            # In some statsmodels versions, params/bse/tvalues/pvalues on a robust
            # wrapper may be a plain numpy array rather than a named Series.
            # Converting to numpy arrays before indexing avoids KeyError / IndexError
            # across statsmodels 0.11–0.14+.
            rb_params_arr = np.asarray(robust_result.params)
            rb_bse_arr = np.asarray(robust_result.bse)
            rb_tvals_arr = np.asarray(robust_result.tvalues)
            rb_pvals_arr = np.asarray(robust_result.pvalues)
            rb_ci_arr = np.asarray(robust_result.conf_int())  # shape (n_params, 2)

            for i, name in enumerate(coef_names):
                label = "Intercept" if name == "const" else name
                out.append(
                    f"  {label:<22} {rb_params_arr[i]:>12.4f} {rb_bse_arr[i]:>12.4f} "
                    f"{rb_tvals_arr[i]:>10.4f} {format_p_value(rb_pvals_arr[i]):>12} "
                    f"{rb_ci_arr[i, 0]:>14.4f} {rb_ci_arr[i, 1]:>14.4f}"
                )
        except Exception as e:
            out.append(f"  [WARNING] Could not display robust results: {e}")
        _dual_print(out, file_handle); out = []

    # ── ASSUMPTION DIAGNOSTICS ───────────────────────────────────────────────
    section_num = 6 if homosced_res["heteroscedastic"] else 5
    out.append(_section(f"{section_num}. ASSUMPTION DIAGNOSTICS"))

    # A. Linearity
    out.append("  [A] LINEARITY — Ramsey RESET Test (power=3: tests Ŷ² and Ŷ³)")
    out.append(f"      F-statistic : {reset_res['F_statistic']:.4f}" if not np.isnan(reset_res['F_statistic']) else "      F-statistic : N/A")
    out.append(f"      p-value     : {format_p_value(reset_res['p_value'])}")
    out.append(f"      Result      : {reset_res['interpretation']}")
    out.append("")

    # B. Residual Normality
    out.append(f"  [B] RESIDUAL NORMALITY — {normality_res['test']}")
    out.append(f"      Statistic   : {normality_res['statistic']:.4f}" if not np.isnan(normality_res['statistic']) else "      Statistic   : N/A")
    out.append(f"      p-value     : {format_p_value(normality_res['p_value'])}")
    out.append(f"      Result      : {normality_res['interpretation']}")
    out.append("")

    # C. Homoscedasticity
    out.append("  [C] HOMOSCEDASTICITY — Breusch-Pagan Test")
    out.append(f"      LM Statistic : {homosced_res['LM_statistic']:.4f}" if not np.isnan(homosced_res['LM_statistic']) else "      LM Statistic : N/A")
    out.append(f"      p-value      : {format_p_value(homosced_res['LM_p_value'])}")
    out.append(f"      Result       : {homosced_res['interpretation']}")
    out.append("")

    # D. Independence
    out.append("  [D] INDEPENDENCE OF ERRORS — Durbin-Watson")
    out.append(f"      DW Statistic : {dw_res['DW_statistic']:.4f}" if not np.isnan(dw_res['DW_statistic']) else "      DW Statistic : N/A")
    out.append(f"      Result       : {dw_res['interpretation']}")
    out.append(f"      {dw_res['caveat']}")
    out.append("")

    # E. Outlier Detection — two tiers
    out.append("  [E] OUTLIER DETECTION — Studentized Residuals (two-tier)")
    out.append(f"      Strong outliers  (|SR| > 3)       : {diag_res['n_outliers']}")
    out.append(f"      Moderate concern (2 < |SR| ≤ 3)   : {diag_res['n_moderate_outliers']}")
    if diag_res["n_outliers"] > 0:
        out.append(f"      Strong outlier indices (0-based)  : {list(diag_res['outlier_indices'])}")
        out.append(
            "      ⚠ STRONG OUTLIERS DETECTED: These observations fall beyond 3 studentized SDs. "
            "Verify clinical data accuracy for each flagged case. "
            "Do not remove without documented clinical justification."
        )
    if diag_res["n_moderate_outliers"] > 0:
        out.append(f"      Moderate concern indices (0-based): {list(diag_res['moderate_outlier_indices'])}")
        out.append(
            "      Moderate concern: These observations fall in the 2–3 studentized SD range. "
            "Inspect individually — they are unusual but not definitive outliers."
        )
    if diag_res["n_outliers"] == 0 and diag_res["n_moderate_outliers"] == 0:
        out.append("      ✓ No outliers detected at either threshold (|SR| > 2 or |SR| > 3).")
    out.append("")

    # F. Leverage
    out.append(f"  [F] LEVERAGE — Hat Values (threshold = 2(k+1)/n = {diag_res['leverage_cutoff']:.4f})")
    out.append(f"      High-leverage observations : {diag_res['n_high_leverage']}")
    if diag_res["n_high_leverage"] > 0:
        out.append(f"      Observation indices (0-based): {list(diag_res['high_leverage_indices'])}")
        out.append(
            "      High-leverage points have unusual predictor patterns and may "
            "disproportionately influence the model."
        )
    else:
        out.append("      ✓ No high-leverage observations detected.")
    out.append("")

    # G. Influence
    out.append(f"  [G] INFLUENCE — Cook's Distance (threshold = 4/n = {diag_res['cooks_cutoff']:.4f})")
    out.append(f"      Influential observations : {diag_res['n_influential']}")
    if diag_res["n_influential"] > 0:
        out.append(f"      Observation indices (0-based): {list(diag_res['influential_indices'])}")
        out.append(
            "      These observations substantially alter the regression coefficients "
            "when excluded. Verify clinical data accuracy."
        )
    else:
        out.append("      ✓ No highly influential observations detected.")
    _dual_print(out, file_handle); out = []

    # ── MEDICAL & PRACTICAL INTERPRETATION ───────────────────────────────────
    section_num += 1
    out.append(_section(f"{section_num}. MEDICAL & PRACTICAL INTERPRETATION"))

    out.append(f"  Outcome Variable: {outcome}")
    out.append(f"  The model explains {100*r2:.2f}% of variability in {outcome}.")
    out.append(f"  Adjusted R² = {r2_adj:.4f} accounts for model complexity.")
    out.append("")
    out.append("  Predictor Summary (statistically significant at p < 0.05):")
    significant_preds = [p for p in predictors if p_vals.get(p, 1.0) < 0.05]
    non_significant_preds = [p for p in predictors if p_vals.get(p, 1.0) >= 0.05]

    if significant_preds:
        for pred in significant_preds:
            b = coefs[pred]
            out.append(f"    ✓ {pred}: β = {b:.4f} (p {format_p_value(p_vals[pred])})")
    else:
        out.append("    No individual predictors reached statistical significance at p < 0.05.")

    if non_significant_preds:
        out.append(f"\n  Non-significant predictors (p ≥ 0.05): {', '.join(non_significant_preds)}")
        out.append("  These may lack power, be confounded, or genuinely have no independent effect.")

    out.append("")
    out.append("  Statistical vs Clinical Significance:")
    out.append("  A predictor may be statistically significant but clinically trivial,")
    out.append("  or clinically important but fail significance due to small sample size.")
    out.append("  Always interpret coefficient magnitude and confidence intervals alongside p-values.")

    out.append("")
    out.append("  Effect Direction and Magnitude:")
    for pred in predictors:
        if pred not in coefs:
            continue
        b = coefs[pred]
        direction = "positive" if b > 0 else "negative"
        out.append(f"    {pred}: {direction} association with {outcome} (β = {b:.4f})")
    _dual_print(out, file_handle); out = []

    # ── LIMITATIONS ───────────────────────────────────────────────────────────
    section_num += 1
    out.append(_section(f"{section_num}. LIMITATIONS"))
    out.append("""
  1. CORRELATION IS NOT CAUSATION
     Regression identifies associations. Causal inference requires study design
     (RCT, propensity score matching, instrumental variables).

  2. OMITTED VARIABLE BIAS
     Unmeasured confounders may distort coefficient estimates. The model's
     validity depends on inclusion of all relevant predictors.

  3. MULTICOLLINEARITY
     Highly correlated predictors produce unstable, unreliable coefficient
     estimates. Individual βs may be misleading even when the overall model is valid.

  4. OUTLIER SENSITIVITY
     OLS is sensitive to extreme values. Influential observations should be
     examined clinically before any exclusion.

  5. ASSUMPTION VIOLATIONS
     If linearity, normality, or homoscedasticity assumptions fail, OLS
     estimates may be biased or inference invalid. Consider transformations
     or alternative models (e.g., generalised linear models).

  6. EXTRAPOLATION RISK
     Model predictions should not be extended beyond the range of observed
     predictor values. Extrapolation is scientifically unjustifiable.

  7. SAMPLE SIZE
     Small samples produce wide confidence intervals and unstable estimates.
     Replication in larger, independent datasets is recommended.

  8. CROSS-SECTIONAL LIMITATIONS
     This analysis does not establish temporal precedence. Longitudinal data
     is needed to make claims about change over time.
    """)
    _dual_print(out, file_handle); out = []

    # ── FOOTER ───────────────────────────────────────────────────────────────
    out.append(_separator())
    out.append("END OF REPORT — VM-WIT-STATS | VM Medical College")
    out.append("For academic publication, please verify all assumptions are met.")
    out.append("This output is research-grade and intended for trained professionals.")
    out.append(_separator())
    _dual_print(out, file_handle)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Main entry point for Multiple Linear Regression analysis.

    Workflow:
        1. Load CSV
        2. Display columns
        3. Select outcome (Y)
        4. Select predictors (X₁, X₂, ..., Xₖ)
        5. Validate
        6. Clean data
        7. Descriptive statistics
        8. Fit OLS model
        9. Diagnostics (VIF, correlation matrix, RESET, Shapiro-Wilk/Lilliefors,
           Breusch-Pagan, DW, two-tier outliers, leverage, influence)
        10. Robust SE if heteroscedasticity detected
        11. Generate and save report
    """
    print(_separator())
    print("VM-WIT-STATS | VM Medical College")
    print("MULTIPLE LINEAR REGRESSION ANALYSIS")
    print(_separator())

    # ── SAMPLE DATA OPTION ────────────────────────────────────────────────────
    use_sample = input(
        "\nGenerate and use a sample medical dataset for testing? [y/n]: "
    ).strip().lower()

    if use_sample == "y":
        filepath = generate_sample_csv()
    else:
        filepath = input("Enter the full path to your CSV file: ").strip()

    # ── LOAD ──────────────────────────────────────────────────────────────────
    df_raw = load_csv(filepath)
    if df_raw is None:
        print("[FATAL] Could not load data. Exiting.")
        sys.exit(1)

    # ── DISPLAY COLUMNS ───────────────────────────────────────────────────────
    display_columns(df_raw)

    # ── SELECT OUTCOME ────────────────────────────────────────────────────────
    outcome = select_outcome(df_raw)
    if outcome is None:
        print("[FATAL] No outcome variable selected. Exiting.")
        sys.exit(1)

    # ── SELECT PREDICTORS ─────────────────────────────────────────────────────
    predictors = select_predictors(df_raw, outcome)
    if predictors is None or len(predictors) < 2:
        print("[FATAL] Insufficient predictors selected. Exiting.")
        sys.exit(1)

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    if not validate_variables(df_raw, outcome, predictors):
        print("[FATAL] Validation failed. Exiting.")
        sys.exit(1)

    # ── CLEAN ─────────────────────────────────────────────────────────────────
    df = clean_data(df_raw, outcome, predictors)
    n = len(df)
    k = len(predictors)

    # ── DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────
    desc_stats = descriptive_statistics(df, outcome, predictors)

    # ── FIT MODEL ─────────────────────────────────────────────────────────────
    result, X_const, y = fit_ols_model(df, outcome, predictors)
    if result is None:
        print("[FATAL] Model fitting failed. Exiting.")
        sys.exit(1)

    # ── DIAGNOSTICS ───────────────────────────────────────────────────────────
    vif_df = compute_vif(df, predictors)
    reset_res = check_linearity_reset(result)
    normality_res = check_residual_normality(result.resid.values)
    homosced_res = check_homoscedasticity(result, X_const, y)
    dw_res = check_independence_dw(result.resid.values)
    diag_res = detect_outliers_leverage_influence(result, X_const, n, k)

    # ── ROBUST SE (if heteroscedastic) ────────────────────────────────────────
    robust_result = None
    if homosced_res["heteroscedastic"]:
        print("\n[INFO] Heteroscedasticity detected. Computing HC3 robust standard errors...")
        robust_result = fit_robust_model(result, X_const, y)

    # ── OUTPUT DIRECTORY ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── GENERATE REPORT ───────────────────────────────────────────────────────
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            generate_report(
                df=df,
                outcome=outcome,
                predictors=predictors,
                desc_stats=desc_stats,
                result=result,
                robust_result=robust_result,
                vif_df=vif_df,
                reset_res=reset_res,
                normality_res=normality_res,
                homosced_res=homosced_res,
                dw_res=dw_res,
                diag_res=diag_res,
                file_handle=fh,
            )
        print(f"\n[INFO] Report saved to: {OUTPUT_FILE}")
    except OSError as e:
        print(f"[ERROR] Could not save report: {e}")
        print("[INFO] Displaying report to console only.")
        import io
        generate_report(
            df=df, outcome=outcome, predictors=predictors, desc_stats=desc_stats,
            result=result, robust_result=robust_result, vif_df=vif_df,
            reset_res=reset_res, normality_res=normality_res,
            homosced_res=homosced_res, dw_res=dw_res, diag_res=diag_res,
            file_handle=io.StringIO(),
        )


if __name__ == "__main__":
    main()