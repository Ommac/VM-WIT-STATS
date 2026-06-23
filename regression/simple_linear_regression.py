# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Simple Linear Regression
# File        : regression/simple_linear_regression.py
# Description : Models the linear relationship between one continuous
#               predictor (X) and one continuous outcome (Y).
#               Y = β₀ + β₁X
#               Production-grade, research-grade, publication-grade.
#               Uses statsmodels OLS with full diagnostic suite.
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import shapiro, t as t_dist

try:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import (
        het_breuschpagan,
        linear_reset,
    )
    from statsmodels.stats.outliers_influence import OLSInfluence
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ======================================================================
# CONSTANTS
# ======================================================================

SEPARATOR  = "=" * 70
SEPARATOR2 = "-" * 70
REPORT_DIR = "outputs"
REPORT_FILE = os.path.join(REPORT_DIR, "simple_linear_regression_result.txt")

HEADER = """
======================================================================
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Simple Linear Regression
======================================================================
"""

LARGE_SAMPLE_THRESHOLD = 200
MIN_SAMPLE_SIZE        = 10

# ======================================================================
# REPORT BUFFER
# ======================================================================

class ReportBuffer:
    """
    Accumulates all console output so the saved report is identical
    to what is printed. Both outputs are written simultaneously.
    """
    def __init__(self):
        self._lines: list[str] = []

    def write(self, text: str):
        print(text)
        self._lines.append(text)

    def get(self) -> str:
        return "\n".join(self._lines)


_buf = ReportBuffer()


def rprint(text: str = ""):
    """Print to console and accumulate in report buffer."""
    _buf.write(text)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_header():
    rprint(HEADER)


def print_section(title: str):
    rprint(f"\n{SEPARATOR2}")
    rprint(f"  {title}")
    rprint(SEPARATOR2)


def fmt(val, decimals: int = 4) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, float) and math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        return f"{val:,.{decimals}f}"
    return str(val)


def is_effectively_constant(arr: np.ndarray, tol: float = 1e-10) -> bool:
    return bool((np.nanmax(arr) - np.nanmin(arr)) < tol)


def check_statsmodels() -> bool:
    if not STATSMODELS_AVAILABLE:
        print("\n  [ERROR] Required package 'statsmodels' is not installed.")
        print("          pip install statsmodels")
        return False
    return True


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_slr.csv",
                         n: int = 80,
                         seed: int = 42):
    """
    Generate a synthetic medical dataset for Simple Linear Regression.

    Columns:
        Patient_ID   : unique identifier
        Age          : predictor X  — patient age (years)
        SystolicBP   : outcome   Y  — systolic blood pressure (mmHg)

    Relationship: SystolicBP ≈ 95 + 0.6 × Age + noise

    Usage:
        from regression.simple_linear_regression import generate_sample_csv
        generate_sample_csv("data/sample_slr.csv")
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(low=25, high=80, size=n).round(1)
    bp  = 95.0 + 0.6 * age + rng.normal(0, 10, size=n)
    bp  = bp.clip(90, 200).round(1)

    df = pd.DataFrame({
        "Patient_ID": [f"PT{str(i + 1).zfill(3)}" for i in range(n)],
        "Age"       : age,
        "SystolicBP": bp,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       Rows: {n}  |  Columns: Patient_ID, Age (X), SystolicBP (Y)")
    print(f"       True relationship: SystolicBP ≈ 95 + 0.6 × Age + noise")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    rprint("\n  TIP: To generate a sample dataset for testing, run:")
    rprint("       from regression.simple_linear_regression import generate_sample_csv")
    rprint("       generate_sample_csv()")

    while True:
        path = input("\n  Enter path to CSV file: ").strip()
        if not path:
            print("  [ERROR] No path entered.")
            continue
        if not path.lower().endswith(".csv"):
            print("  [WARNING] File does not have a .csv extension. Attempting to load.")
        if not os.path.isfile(path):
            print(f"  [ERROR] File not found: '{path}'.")
            continue
        return path


def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        print("  [ERROR] CSV file is empty.")
        return None
    except pd.errors.ParserError as exc:
        print(f"  [ERROR] Failed to parse CSV.\n          {exc}")
        return None
    except PermissionError:
        print("  [ERROR] Permission denied.")
        return None
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return None

    if df.empty:
        print("  [ERROR] Dataset has no rows.")
        return None

    rprint(f"\n  [OK] Dataset loaded.")
    rprint(f"       Rows    : {df.shape[0]:,}")
    rprint(f"       Columns : {df.shape[1]:,}")
    return df


def display_columns(df: pd.DataFrame):
    print_section("STEP 2 | AVAILABLE COLUMNS")
    rprint(f"\n  {'#':<5} {'Column Name':<40} {'Dtype':<15}")
    rprint(f"  {'-'*5} {'-'*40} {'-'*15}")
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        rprint(f"  {idx:<5} {col:<40} {str(dtype):<15}")


# ======================================================================
# COLUMN SELECTION
# ======================================================================

def select_numeric_column(df: pd.DataFrame,
                           label: str,
                           exclude: str | None = None) -> str | None:
    rprint(f"\n  Select the {label} column (must be continuous and numeric).")
    while True:
        col = input(f"\n  Enter {label} column name: ").strip()
        if not col:
            print("  [ERROR] No column name entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue
        if exclude and col == exclude:
            print(f"  [ERROR] Cannot use '{col}' for both X and Y.")
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        valid = int(coerced.notna().sum())
        if valid == 0:
            print(f"  [ERROR] '{col}' has no numeric values.")
            continue
        non_num = len(coerced) - valid
        if non_num > 0:
            print(f"  [WARNING] {non_num:,} non-numeric value(s) will be treated as missing.")
        rprint(f"  [OK] {label} column selected: '{col}'  (valid n = {valid:,})")
        return col


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_data(df: pd.DataFrame,
               x_col: str,
               y_col: str) -> pd.DataFrame | None:
    print_section("STEP 5 | DATA CLEANING & VALIDATION")

    working = df[[x_col, y_col]].copy()
    orig_n = len(working)
    rprint(f"\n  Original rows : {orig_n:,}")

    working[x_col] = pd.to_numeric(working[x_col], errors="coerce")
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")

    before = len(working)
    working = working.dropna(subset=[x_col, y_col])
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        rprint(f"  Dropped (NaN)       : {dropped_nan:,} row(s)")

    before = len(working)
    inf_mask = (np.isinf(working[x_col]) | np.isinf(working[y_col]))
    working = working[~inf_mask]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        rprint(f"  Dropped (Infinite)  : {dropped_inf:,} row(s)")

    n = len(working)
    rprint(f"\n  Total dropped   : {orig_n - n:,}")
    rprint(f"  Rows for analysis: {n:,}")

    if n == 0:
        rprint("\n  [ERROR] No valid rows remain.")
        return None
    if n < MIN_SAMPLE_SIZE:
        rprint(f"\n  [ERROR] Only {n} observations remain.")
        rprint(f"          Simple Linear Regression requires n ≥ {MIN_SAMPLE_SIZE}.")
        return None

    x_arr = working[x_col].to_numpy(dtype=float)
    y_arr = working[y_col].to_numpy(dtype=float)

    if is_effectively_constant(x_arr):
        rprint(f"\n  [ERROR] Predictor '{x_col}' has zero variance.")
        rprint("          Regression cannot be fitted to a constant predictor.")
        return None
    if is_effectively_constant(y_arr):
        rprint(f"\n  [WARNING] Outcome '{y_col}' has zero variance.")
        rprint("            R² will be undefined. Proceeding with caution.")

    if n < 30:
        rprint(f"\n  [WARNING] Small sample (n = {n}). Estimates may be unstable.")

    rprint(f"\n  [OK] Data is clean and ready for regression.")
    return working.reset_index(drop=True)


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def compute_desc(arr: np.ndarray) -> dict:
    n = len(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n"       : n,
        "mean"    : float(np.mean(arr)),
        "median"  : float(np.median(arr)),
        "std"     : float(np.std(arr, ddof=1)) if n > 1 else float("nan"),
        "sem"     : float(stats.sem(arr))       if n > 1 else float("nan"),
        "min"     : float(np.min(arr)),
        "max"     : float(np.max(arr)),
        "q1"      : float(q1),
        "q3"      : float(q3),
        "iqr"     : float(q3 - q1),
        "skewness": float(stats.skew(arr))      if n > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(arr))  if n > 3 else float("nan"),
    }


def display_descriptive_stats(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    print_section("DESCRIPTIVE STATISTICS")

    x_arr = df[x_col].to_numpy(dtype=float)
    y_arr = df[y_col].to_numpy(dtype=float)
    desc  = {x_col: compute_desc(x_arr), y_col: compute_desc(y_arr)}

    stat_rows = [
        ("Sample Size (n)",   "n"),
        ("Mean",              "mean"),
        ("Median",            "median"),
        ("Std Deviation",     "std"),
        ("Std Error of Mean", "sem"),
        ("Minimum",           "min"),
        ("Maximum",           "max"),
        ("25th Percentile",   "q1"),
        ("75th Percentile",   "q3"),
        ("IQR (Q3 - Q1)",     "iqr"),
        ("Skewness",          "skewness"),
        ("Kurtosis",          "kurtosis"),
    ]

    lw = 28
    cw = 20
    rprint(f"\n  {'Statistic':<{lw}} {x_col:>{cw}} {y_col:>{cw}}")
    rprint(f"  {'-'*lw} {'-'*cw} {'-'*cw}")
    for name, key in stat_rows:
        rprint(f"  {name:<{lw}} {fmt(desc[x_col][key]):>{cw}} {fmt(desc[y_col][key]):>{cw}}")

    # Correlation
    r_pearson, p_pearson = stats.pearsonr(x_arr, y_arr)
    r_spearman, p_spearman = stats.spearmanr(x_arr, y_arr)
    rprint(f"\n  Pearson  r  : {r_pearson:.4f}  (p = {p_pearson:.4f})")
    rprint(f"  Spearman ρ  : {r_spearman:.4f}  (p = {p_spearman:.4f})")
    rprint(f"\n  NOTE: Correlation coefficients are descriptive only.")
    rprint(f"        Linearity is formally assessed by Ramsey RESET Test (see diagnostics).")

    return desc


# ======================================================================
# MODEL FITTING
# ======================================================================

def fit_ols(df: pd.DataFrame,
            x_col: str,
            y_col: str):
    """
    Fit Ordinary Least Squares regression using statsmodels.

    Returns the fitted OLS results object.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    X = sm.add_constant(x)   # adds intercept column
    model = sm.OLS(y, X)

    try:
        result = model.fit()
    except Exception as exc:
        rprint(f"  [ERROR] OLS fitting failed.\n          {exc}")
        return None

    return result


def display_model_results(result, x_col: str, y_col: str):
    print_section("REGRESSION MODEL RESULTS")

    intercept = result.params[0]
    slope     = result.params[1]
    se_int    = result.bse[0]
    se_slope  = result.bse[1]
    t_int     = result.tvalues[0]
    t_slope   = result.tvalues[1]
    p_int     = result.pvalues[0]
    p_slope   = result.pvalues[1]
    ci        = result.conf_int(alpha=0.05)
    ci_int    = ci[0]
    ci_slope  = ci[1]

    rprint(f"\n  Model: {y_col} = β₀ + β₁ × {x_col}")
    rprint(f"  Model: {y_col} = {intercept:.4f} + {slope:.4f} × {x_col}\n")

    rprint(f"  {SEPARATOR2[:68]}")
    rprint(f"  {'Parameter':<28} {'Estimate':>12} {'SE':>10} {'t':>10} {'p-value':>10}")
    rprint(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    rprint(f"  {'Intercept (β₀)':<28} {intercept:>12.4f} {se_int:>10.4f} "
           f"{t_int:>10.4f} {p_int:>10.4f}")
    rprint(f"  {'Slope (β₁) — ' + x_col:<28} {slope:>12.4f} {se_slope:>10.4f} "
           f"{t_slope:>10.4f} {p_slope:>10.4f}")
    rprint(f"  {SEPARATOR2[:68]}")

    rprint(f"\n  95% Confidence Intervals:")
    rprint(f"  {'Intercept (β₀)':<28} [{ci_int[0]:>10.4f},  {ci_int[1]:>10.4f}]")
    rprint(f"  {'Slope (β₁)':<28} [{ci_slope[0]:>10.4f},  {ci_slope[1]:>10.4f}]")

    rprint(f"\n  Slope Interpretation:")
    rprint(f"  For each 1-unit increase in '{x_col}', '{y_col}' changes by")
    rprint(f"  {slope:.4f} units on average (95% CI: {ci_slope[0]:.4f} to {ci_slope[1]:.4f}).")
    if p_slope < 0.05:
        rprint(f"  The slope is statistically significant (p = {p_slope:.4f}).")
    else:
        rprint(f"  The slope is NOT statistically significant (p = {p_slope:.4f}).")


def display_model_performance(result):
    print_section("MODEL PERFORMANCE")

    n   = int(result.nobs)
    r2  = result.rsquared
    ar2 = result.rsquared_adj
    f   = result.fvalue
    fp  = result.f_pvalue
    mse = result.mse_resid
    rse = math.sqrt(mse) if mse >= 0 else float("nan")
    aic = result.aic
    bic = result.bic
    df_model = int(result.df_model)
    df_resid = int(result.df_resid)

    rprint(f"\n  {'Metric':<40} {'Value':>20}")
    rprint(f"  {'-'*40} {'-'*20}")
    rprint(f"  {'R² (Coefficient of Determination)':<40} {r2:>20.4f}")
    rprint(f"  {'Adjusted R²':<40} {ar2:>20.4f}")
    rprint(f"  {'F Statistic':<40} {f:>20.4f}")
    rprint(f"  {'Model p-value':<40} {fp:>20.4f}")
    rprint(f"  {'Residual Standard Error (RSE)':<40} {rse:>20.4f}")
    rprint(f"  {'AIC':<40} {aic:>20.4f}")
    rprint(f"  {'BIC':<40} {bic:>20.4f}")
    rprint(f"  {'Degrees of Freedom (Model)':<40} {df_model:>20,}")
    rprint(f"  {'Degrees of Freedom (Residual)':<40} {df_resid:>20,}")
    rprint(f"  {'Sample Size (n)':<40} {n:>20,}")

    rprint(f"\n  R² Interpretation:")
    rprint(f"  {r2*100:.1f}% of the variance in the outcome is explained by the predictor.")
    if r2 < 0.10:
        rprint(f"  This is a very low R². The predictor explains little of the outcome variance.")
    elif r2 < 0.30:
        rprint(f"  This is a low R². Modest explanatory power.")
    elif r2 < 0.50:
        rprint(f"  This is a moderate R².")
    elif r2 < 0.70:
        rprint(f"  This is a moderately high R².")
    else:
        rprint(f"  This is a high R². Strong explanatory power.")

    rprint(f"\n  RSE = {rse:.4f}: average deviation of observed values from the fitted line.")

    if n >= LARGE_SAMPLE_THRESHOLD and fp < 0.05:
        rprint(f"\n  [LARGE SAMPLE NOTE] n = {n:,}. With large samples, even trivially")
        rprint(f"  small slopes can achieve statistical significance. Assess R² and")
        rprint(f"  slope magnitude for practical significance, not p-value alone.")


# ======================================================================
# REGRESSION DIAGNOSTICS
# ======================================================================

def run_diagnostics(result, df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Run the full diagnostic suite for OLS regression:
      1. Linearity         — Ramsey RESET Test
      2. Residual normality — Shapiro-Wilk on residuals
      3. Homoscedasticity  — Breusch-Pagan Test
      4. Independence      — Durbin-Watson Statistic
      5. Outliers          — Studentized Residuals (|SR| > 3)
      6. Leverage          — Hat values > 2(p+1)/n
      7. Influence         — Cook's Distance > 4/n
    Returns a dictionary of all diagnostic results.
    """
    n = int(result.nobs)
    p = 1   # number of predictors (simple regression)
    residuals = result.resid.to_numpy(dtype=float)
    fitted    = result.fittedvalues.to_numpy(dtype=float)

    diag = {}

    # ------------------------------------------------------------------
    # 1. LINEARITY — Ramsey RESET Test
    #    Tests whether higher-order terms of fitted values improve the model.
    #    H0: model is correctly specified (linear).
    #    H1: model misspecification (nonlinearity).
    # ------------------------------------------------------------------
    try:
        reset_result = linear_reset(result, power=2, use_f=True)
        diag["reset_f"]   = float(reset_result.fvalue)
        diag["reset_p"]   = float(reset_result.pvalue)
        diag["linear_ok"] = bool(reset_result.pvalue > 0.05)
    except Exception as exc:
        diag["reset_f"]   = float("nan")
        diag["reset_p"]   = float("nan")
        diag["linear_ok"] = None
        diag["reset_err"] = str(exc)

    # ------------------------------------------------------------------
    # 2. RESIDUAL NORMALITY — Shapiro-Wilk
    # ------------------------------------------------------------------
    if n < 3:
        diag["sw_stat"]    = float("nan")
        diag["sw_p"]       = float("nan")
        diag["resid_norm"] = None
        diag["sw_note"]    = "n < 3: cannot run Shapiro-Wilk"
    elif n > 5000:
        diag["sw_stat"]    = float("nan")
        diag["sw_p"]       = float("nan")
        diag["resid_norm"] = None
        diag["sw_note"]    = "n > 5000: Shapiro-Wilk not computed"
    else:
        sw_stat, sw_p = shapiro(residuals)
        diag["sw_stat"]    = float(sw_stat)
        diag["sw_p"]       = float(sw_p)
        diag["resid_norm"] = bool(sw_p > 0.05)
        diag["sw_note"]    = ""

    # ------------------------------------------------------------------
    # 3. HOMOSCEDASTICITY — Breusch-Pagan Test
    #    H0: residuals have constant variance.
    #    H1: residual variance depends on X (heteroscedasticity).
    # ------------------------------------------------------------------
    try:
        X_with_const = sm.add_constant(df[x_col].to_numpy(dtype=float))
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_with_const)
        diag["bp_lm"]      = float(bp_lm)
        diag["bp_lm_p"]    = float(bp_lm_p)
        diag["bp_f"]       = float(bp_f)
        diag["bp_f_p"]     = float(bp_f_p)
        diag["homosced"]   = bool(bp_lm_p > 0.05)
    except Exception as exc:
        diag["bp_lm"]      = float("nan")
        diag["bp_lm_p"]    = float("nan")
        diag["bp_f"]       = float("nan")
        diag["bp_f_p"]     = float("nan")
        diag["homosced"]   = None
        diag["bp_err"]     = str(exc)

    # ------------------------------------------------------------------
    # 4. INDEPENDENCE — Durbin-Watson Statistic
    #    Values near 2 indicate no autocorrelation.
    #    Values < 1.5 or > 2.5 suggest autocorrelation.
    # ------------------------------------------------------------------
    try:
        dw = float(durbin_watson(residuals))
        diag["dw"]          = dw
        diag["dw_ok"]       = bool(1.5 <= dw <= 2.5)
    except Exception:
        diag["dw"]          = float("nan")
        diag["dw_ok"]       = None

    # ------------------------------------------------------------------
    # 5. OUTLIERS — Studentized Residuals
    #    |SR| > 3 flags an observation as a potential outlier.
    # ------------------------------------------------------------------
    try:
        influence    = OLSInfluence(result)
        stud_resid   = influence.resid_studentized_external
        outlier_mask = np.abs(stud_resid) > 3.0
        outlier_idx  = np.where(outlier_mask)[0]
        diag["stud_resid"]     = stud_resid
        diag["outlier_idx"]    = outlier_idx.tolist()
        diag["n_outliers"]     = int(outlier_mask.sum())
    except Exception as exc:
        diag["stud_resid"]     = np.full(n, float("nan"))
        diag["outlier_idx"]    = []
        diag["n_outliers"]     = 0
        diag["outlier_err"]    = str(exc)

    # ------------------------------------------------------------------
    # 6. LEVERAGE — Hat Values
    #    Threshold: h > 2(p+1)/n
    # ------------------------------------------------------------------
    try:
        hat_values      = influence.hat_matrix_diag
        lev_threshold   = 2.0 * (p + 1) / n
        high_lev_mask   = hat_values > lev_threshold
        high_lev_idx    = np.where(high_lev_mask)[0]
        diag["hat_values"]     = hat_values
        diag["lev_threshold"]  = float(lev_threshold)
        diag["high_lev_idx"]   = high_lev_idx.tolist()
        diag["n_high_lev"]     = int(high_lev_mask.sum())
    except Exception as exc:
        diag["hat_values"]     = np.full(n, float("nan"))
        diag["lev_threshold"]  = float("nan")
        diag["high_lev_idx"]   = []
        diag["n_high_lev"]     = 0
        diag["lev_err"]        = str(exc)

    # ------------------------------------------------------------------
    # 7. INFLUENCE — Cook's Distance
    #    Threshold: D > 4/n
    # ------------------------------------------------------------------
    try:
        cooks_d, _       = influence.cooks_distance
        cook_threshold   = 4.0 / n
        high_cook_mask   = cooks_d > cook_threshold
        high_cook_idx    = np.where(high_cook_mask)[0]
        diag["cooks_d"]         = cooks_d
        diag["cook_threshold"]  = float(cook_threshold)
        diag["high_cook_idx"]   = high_cook_idx.tolist()
        diag["n_high_cook"]     = int(high_cook_mask.sum())
    except Exception as exc:
        diag["cooks_d"]         = np.full(n, float("nan"))
        diag["cook_threshold"]  = float("nan")
        diag["high_cook_idx"]   = []
        diag["n_high_cook"]     = 0
        diag["cook_err"]        = str(exc)

    return diag


def display_diagnostics(diag: dict, n: int):
    print_section("REGRESSION DIAGNOSTICS")

    # --- 1. Linearity ---
    rprint("\n  [1] LINEARITY — Ramsey RESET Test")
    rprint(f"      H0: Model is correctly specified (linear relationship).")
    rprint(f"      H1: Model is misspecified (non-linear pattern present).")
    if math.isnan(diag.get("reset_f", float("nan"))):
        note = diag.get("reset_err", "RESET test could not be computed.")
        rprint(f"      [NOTE] {note}")
    else:
        verdict = "PASS — Linearity assumption supported" \
                  if diag["linear_ok"] else \
                  "FAIL — Evidence of non-linearity. Consider transformations."
        rprint(f"      F Statistic : {diag['reset_f']:.4f}")
        rprint(f"      p-value     : {diag['reset_p']:.4f}")
        rprint(f"      Result      : {verdict}")

    # --- 2. Residual Normality ---
    rprint("\n  [2] RESIDUAL NORMALITY — Shapiro-Wilk Test on Residuals")
    rprint(f"      H0: Residuals are normally distributed.")
    rprint(f"      H1: Residuals are NOT normally distributed.")
    if diag.get("sw_note"):
        rprint(f"      [NOTE] {diag['sw_note']}")
    else:
        verdict_norm = "Normal" if bool(diag["resid_norm"]) else "Non-Normal"
        verdict_str  = ("PASS — Residual normality supported"
                        if bool(diag["resid_norm"])
                        else "FAIL — Evidence against normality. Inference may be affected.")
        rprint(f"      W Statistic : {diag['sw_stat']:.4f}")
        rprint(f"      p-value     : {diag['sw_p']:.4f}")
        rprint(f"      Residuals   : {verdict_norm}")
        rprint(f"      Result      : {verdict_str}")
    if n >= LARGE_SAMPLE_THRESHOLD:
        rprint(f"      [NOTE] Large sample (n = {n:,}). OLS inference is robust to")
        rprint(f"             non-normal residuals by the Central Limit Theorem.")

    # --- 3. Homoscedasticity ---
    rprint("\n  [3] HOMOSCEDASTICITY — Breusch-Pagan Test")
    rprint(f"      H0: Residuals have constant variance (homoscedastic).")
    rprint(f"      H1: Residual variance depends on X (heteroscedastic).")
    if math.isnan(diag.get("bp_lm", float("nan"))):
        note = diag.get("bp_err", "Breusch-Pagan test could not be computed.")
        rprint(f"      [NOTE] {note}")
    else:
        verdict_bp = ("PASS — Homoscedasticity assumption supported"
                      if diag["homosced"]
                      else "FAIL — Heteroscedasticity detected. Consider robust SE.")
        rprint(f"      LM Statistic : {diag['bp_lm']:.4f}")
        rprint(f"      LM p-value   : {diag['bp_lm_p']:.4f}")
        rprint(f"      F Statistic  : {diag['bp_f']:.4f}")
        rprint(f"      F p-value    : {diag['bp_f_p']:.4f}")
        rprint(f"      Result       : {verdict_bp}")
        if not diag["homosced"]:
            rprint(f"      Action       : Consider heteroscedasticity-robust standard errors,")
            rprint(f"                     variable transformation, or weighted least squares.")

    # --- 4. Independence ---
    rprint("\n  [4] INDEPENDENCE OF ERRORS — Durbin-Watson Statistic")
    rprint(f"      Values near 2.0 indicate no autocorrelation.")
    rprint(f"      Values < 1.5 suggest positive autocorrelation.")
    rprint(f"      Values > 2.5 suggest negative autocorrelation.")
    if math.isnan(diag.get("dw", float("nan"))):
        rprint(f"      [NOTE] Durbin-Watson could not be computed.")
    else:
        dw = diag["dw"]
        if dw < 1.5:
            dw_interp = "Positive autocorrelation suspected."
        elif dw > 2.5:
            dw_interp = "Negative autocorrelation suspected."
        else:
            dw_interp = "No autocorrelation detected."
        verdict_dw = ("PASS — Independence assumption supported"
                      if diag["dw_ok"]
                      else f"CONCERN — {dw_interp}")
        rprint(f"      DW Statistic : {dw:.4f}")
        rprint(f"      Result       : {verdict_dw}")
        rprint(f"      [NOTE] Durbin-Watson primarily detects serial autocorrelation.")
        rprint(f"             It is most relevant for time-ordered observations.")

    # --- 5. Outliers ---
    rprint("\n  [5] OUTLIER DETECTION — Studentized Residuals (threshold: |SR| > 3)")
    n_out = diag["n_outliers"]
    if n_out == 0:
        rprint(f"      No outliers detected (|Studentized Residual| ≤ 3 for all observations).")
    else:
        rprint(f"      Outliers detected: {n_out} observation(s) with |SR| > 3")
        idx_list = [str(i + 1) for i in diag["outlier_idx"][:20]]
        rprint(f"      Row indices: {', '.join(idx_list)}"
               + (" ..." if len(diag["outlier_idx"]) > 20 else ""))
        rprint(f"      [ACTION] Investigate these observations for data entry errors")
        rprint(f"               or clinically extreme values.")

    # --- 6. Leverage ---
    rprint(f"\n  [6] LEVERAGE — Hat Values (threshold: h > 2(p+1)/n = {diag['lev_threshold']:.4f})")
    n_lev = diag["n_high_lev"]
    if n_lev == 0:
        rprint(f"      No high-leverage observations detected.")
    else:
        rprint(f"      High-leverage observations: {n_lev}")
        idx_list = [str(i + 1) for i in diag["high_lev_idx"][:20]]
        rprint(f"      Row indices: {', '.join(idx_list)}"
               + (" ..." if len(diag["high_lev_idx"]) > 20 else ""))
        rprint(f"      [NOTE] High leverage indicates extreme X values.")
        rprint(f"             These observations have high potential to influence the slope.")

    # --- 7. Influence ---
    rprint(f"\n  [7] INFLUENCE — Cook's Distance (threshold: D > 4/n = {diag['cook_threshold']:.4f})")
    n_cook = diag["n_high_cook"]
    if n_cook == 0:
        rprint(f"      No highly influential observations detected.")
    else:
        rprint(f"      Influential observations: {n_cook}")
        idx_list = [str(i + 1) for i in diag["high_cook_idx"][:20]]
        rprint(f"      Row indices: {', '.join(idx_list)}"
               + (" ..." if len(diag["high_cook_idx"]) > 20 else ""))
        rprint(f"      [ACTION] Examine whether these observations disproportionately")
        rprint(f"               affect the regression coefficients.")
        rprint(f"               Consider sensitivity analysis with and without them.")

    # --- Diagnostic Summary ---
    rprint(f"\n  DIAGNOSTIC SUMMARY:")
    checks = []

    lin_ok = diag.get("linear_ok")
    if lin_ok is True:
        checks.append(("Linearity (RESET)",      "PASS"))
    elif lin_ok is False:
        checks.append(("Linearity (RESET)",      "FAIL"))
    else:
        checks.append(("Linearity (RESET)",      "N/A"))

    norm_ok = diag.get("resid_norm")
    if diag.get("sw_note"):
        checks.append(("Residual Normality",     "N/A"))
    elif norm_ok is True:
        checks.append(("Residual Normality",     "PASS"))
    else:
        checks.append(("Residual Normality",     "FAIL"))

    homo_ok = diag.get("homosced")
    if homo_ok is True:
        checks.append(("Homoscedasticity (BP)",  "PASS"))
    elif homo_ok is False:
        checks.append(("Homoscedasticity (BP)",  "FAIL"))
    else:
        checks.append(("Homoscedasticity (BP)",  "N/A"))

    dw_ok = diag.get("dw_ok")
    if dw_ok is True:
        checks.append(("Independence (DW)",      "PASS"))
    elif dw_ok is False:
        checks.append(("Independence (DW)",      "CONCERN"))
    else:
        checks.append(("Independence (DW)",      "N/A"))

    checks.append(("Outliers",                   f"{n_out} flagged"))
    checks.append(("High Leverage",              f"{n_lev} flagged"))
    checks.append(("Influential Obs (Cook's D)", f"{n_cook} flagged"))

    rprint(f"\n  {'Diagnostic':<35} {'Result':>20}")
    rprint(f"  {'-'*35} {'-'*20}")
    for name, res in checks:
        rprint(f"  {name:<35} {res:>20}")


# ======================================================================
# PREDICTION
# ======================================================================

def display_prediction(result, df: pd.DataFrame, x_col: str, y_col: str):
    print_section("PREDICTION INTERVALS")

    x_arr = df[x_col].to_numpy(dtype=float)
    n     = len(x_arr)

    # Compute at: mean, Q1, Q3, min, max of X
    x_vals = {
        "Minimum X"   : float(np.min(x_arr)),
        "Q1 of X"     : float(np.percentile(x_arr, 25)),
        "Mean X"      : float(np.mean(x_arr)),
        "Q3 of X"     : float(np.percentile(x_arr, 75)),
        "Maximum X"   : float(np.max(x_arr)),
    }

    rprint(f"\n  Predictions at key values of '{x_col}':\n")
    rprint(f"  {'X Value':<14} {'X':<12} {'Ŷ (fit)':<12} "
           f"{'Mean 95% CI':<24} {'Pred 95% PI':<24}")
    rprint(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*24} {'-'*24}")

    for label, x_val in x_vals.items():
        X_pred = sm.add_constant(np.array([[1, x_val]]), has_constant='add')
        # Drop the double-constant that add_constant creates when input has 2 cols
        X_pred = np.array([[1.0, x_val]])
        pred   = result.get_prediction(X_pred)
        summary = pred.summary_frame(alpha=0.05)

        y_fit    = float(summary["mean"].iloc[0])
        ci_lo    = float(summary["mean_ci_lower"].iloc[0])
        ci_hi    = float(summary["mean_ci_upper"].iloc[0])
        pi_lo    = float(summary["obs_ci_lower"].iloc[0])
        pi_hi    = float(summary["obs_ci_upper"].iloc[0])

        rprint(f"  {label:<14} {x_val:<12.2f} {y_fit:<12.4f} "
               f"[{ci_lo:>8.4f}, {ci_hi:>8.4f}]  "
               f"[{pi_lo:>8.4f}, {pi_hi:>8.4f}]")

    rprint(f"\n  MEAN CONFIDENCE INTERVAL (CI):")
    rprint(f"  Estimates the range within which the TRUE MEAN response lies")
    rprint(f"  for a given X value, with 95% confidence.")
    rprint(f"  This interval is narrower — it describes the mean of many subjects.")

    rprint(f"\n  PREDICTION INTERVAL (PI):")
    rprint(f"  Estimates the range within which a SINGLE NEW OBSERVATION will fall")
    rprint(f"  for a given X value, with 95% probability.")
    rprint(f"  This interval is wider — it must account for individual variability.")

    rprint(f"\n  EXTRAPOLATION WARNING:")
    rprint(f"  Predictions outside the observed range of '{x_col}'")
    rprint(f"  ({fmt(np.min(x_arr))} to {fmt(np.max(x_arr))}) are extrapolations.")
    rprint(f"  The linear relationship may not hold beyond the observed data range.")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(result,
                            diag: dict,
                            desc: dict,
                            x_col: str,
                            y_col: str):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    n         = int(result.nobs)
    intercept = float(result.params[0])
    slope     = float(result.params[1])
    p_slope   = float(result.pvalues[1])
    r2        = float(result.rsquared)
    ci        = result.conf_int(alpha=0.05)
    ci_slope  = ci[1]
    sig       = p_slope < 0.05

    rprint(f"\n  PREDICTOR  : {x_col}")
    rprint(f"  OUTCOME    : {y_col}")
    rprint(f"  SAMPLE SIZE: {n:,}")

    rprint(f"\n  WHAT DOES SIMPLE LINEAR REGRESSION MEASURE?")
    rprint(f"  Simple Linear Regression models the linear association between")
    rprint(f"  a continuous predictor ({x_col}) and a continuous outcome ({y_col}).")
    rprint(f"  The slope (β₁) quantifies the average change in {y_col} per")
    rprint(f"  1-unit increase in {x_col}, holding other factors constant.")
    rprint(f"  The intercept (β₀) is the predicted {y_col} when {x_col} = 0")
    rprint(f"  (meaningful only if X = 0 is within the data range).")

    rprint(f"\n  STATISTICAL CONCLUSION:")
    if sig:
        rprint(f"  There is a statistically significant linear association between")
        rprint(f"  '{x_col}' and '{y_col}'.")
        rprint(f"  (β₁ = {slope:.4f}, 95% CI [{ci_slope[0]:.4f}, {ci_slope[1]:.4f}], "
               f"p = {p_slope:.4f})")
        rprint(f"\n  For each 1-unit increase in '{x_col}', '{y_col}' changes by")
        rprint(f"  {slope:.4f} units on average.")
    else:
        rprint(f"  No statistically significant linear association was detected between")
        rprint(f"  '{x_col}' and '{y_col}'.")
        rprint(f"  (β₁ = {slope:.4f}, 95% CI [{ci_slope[0]:.4f}, {ci_slope[1]:.4f}], "
               f"p = {p_slope:.4f})")

    rprint(f"\n  STATISTICAL SIGNIFICANCE vs PRACTICAL SIGNIFICANCE:")
    rprint(f"  Statistical significance (p < 0.05) indicates that the observed")
    rprint(f"  association is unlikely due to chance alone. It does NOT indicate")
    rprint(f"  that the association is clinically meaningful or large.")
    rprint(f"\n  R² = {r2:.4f}: The predictor explains {r2*100:.1f}% of the variance")
    rprint(f"  in the outcome. {'This is a weak-to-moderate effect.' if r2 < 0.30 else 'This indicates meaningful explanatory power.' if r2 < 0.70 else 'This is a strong effect.'}")

    if n >= LARGE_SAMPLE_THRESHOLD:
        rprint(f"\n  LARGE-SAMPLE NOTE (n = {n:,}):")
        rprint(f"  With large samples, even very small slopes can achieve")
        rprint(f"  statistical significance. A p-value < 0.05 alone does not")
        rprint(f"  confirm clinical importance. Evaluate the slope magnitude,")
        rprint(f"  confidence interval width, and R² for practical relevance.")

    rprint(f"\n  ASSUMPTION DIAGNOSTICS SUMMARY:")
    fails = []
    if diag.get("linear_ok") is False:
        fails.append("Non-linearity detected (RESET Test failed) — model may be misspecified.")
    if diag.get("resid_norm") is False and not diag.get("sw_note"):
        fails.append("Residual non-normality (Shapiro-Wilk) — inference may be affected in small samples.")
    if diag.get("homosced") is False:
        fails.append("Heteroscedasticity (Breusch-Pagan) — standard errors may be underestimated.")
    if diag.get("dw_ok") is False:
        fails.append("Autocorrelation suspected (Durbin-Watson) — independence assumption at risk.")
    if diag["n_outliers"] > 0:
        fails.append(f"{diag['n_outliers']} outlier(s) detected (|Studentized Residual| > 3).")
    if diag["n_high_cook"] > 0:
        fails.append(f"{diag['n_high_cook']} influential observation(s) detected (Cook's D > 4/n).")

    if not fails:
        rprint(f"  All key assumptions appear to be satisfied.")
        rprint(f"  Regression results are reliable under standard OLS conditions.")
    else:
        rprint(f"  The following assumption issues were detected:")
        for f in fails:
            rprint(f"    • {f}")
        rprint(f"  These issues may affect the reliability of inference.")
        rprint(f"  Address violations before reporting final results.")

    rprint(f"\n  CLINICAL RELEVANCE:")
    rprint(f"  A statistically significant association between '{x_col}' and '{y_col}'")
    rprint(f"  should be interpreted in the context of:")
    rprint(f"    • Clinical plausibility (does the relationship make biological sense?)")
    rprint(f"    • Effect magnitude (is the slope clinically meaningful?)")
    rprint(f"    • Confounding (other variables may explain the association)")
    rprint(f"    • Study design (cross-sectional, observational, or experimental?)")

    rprint(f"\n  LIMITATIONS:")
    rprint(f"  • CORRELATION ≠ CAUSATION. A significant slope does not establish")
    rprint(f"    a causal relationship between {x_col} and {y_col}.")
    rprint(f"  • OUTLIER SENSITIVITY. OLS is sensitive to extreme observations.")
    rprint(f"    {diag['n_outliers']} outlier(s) were detected and may affect results.")
    rprint(f"  • ASSUMPTION VIOLATIONS. OLS efficiency depends on linearity,")
    rprint(f"    normality of residuals, homoscedasticity, and independence.")
    rprint(f"  • EXTRAPOLATION RISK. Predictions outside the observed range of")
    rprint(f"    '{x_col}' are extrapolations. The model may not generalise.")
    rprint(f"  • CONFOUNDING. Simple regression does not adjust for other variables.")
    rprint(f"    Multiple regression should be considered for adjusted estimates.")
    rprint(f"  • SINGLE PREDICTOR. This model uses only '{x_col}'. Unexplained")
    rprint(f"    variance (1 − R² = {(1-r2)*100:.1f}%) is attributed to other factors.")


# ======================================================================
# REPORT SAVE
# ======================================================================

def save_report():
    print_section("SAVE REPORT")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(_buf.get())
        rprint(f"\n  [OK] Report saved successfully.")
        rprint(f"       Path: {os.path.abspath(REPORT_FILE)}")
    except PermissionError:
        print(f"  [ERROR] Permission denied writing to: {REPORT_FILE}")
    except OSError as exc:
        print(f"  [ERROR] Failed to save report.\n          {exc}")


# ======================================================================
# MAIN ORCHESTRATOR
# ======================================================================

def run_simple_linear_regression():
    if not check_statsmodels():
        sys.exit(1)

    print_header()
    rprint("  This module fits a Simple Linear Regression model.")
    rprint("  Model: Y = β₀ + β₁X")
    rprint("  Includes full diagnostic suite and medical interpretation.")

    # Step 1: Load
    path = get_csv_path()
    df = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2: Columns
    display_columns(df)

    # Step 3: Select X
    print_section("STEP 3 | SELECT PREDICTOR VARIABLE (X)")
    x_col = select_numeric_column(df, "Predictor (X)")
    if x_col is None:
        sys.exit(1)

    # Step 4: Select Y
    print_section("STEP 4 | SELECT OUTCOME VARIABLE (Y)")
    y_col = select_numeric_column(df, "Outcome (Y)", exclude=x_col)
    if y_col is None:
        sys.exit(1)

    # Step 5: Clean
    clean_df = clean_data(df, x_col, y_col)
    if clean_df is None:
        sys.exit(1)

    # Step 6: Descriptive
    desc = display_descriptive_stats(clean_df, x_col, y_col)

    # Step 7: Fit model
    print_section("STEP 6 | FITTING OLS MODEL")
    result = fit_ols(clean_df, x_col, y_col)
    if result is None:
        sys.exit(1)
    rprint(f"  [OK] OLS model fitted successfully.")

    # Step 8: Results
    display_model_results(result, x_col, y_col)
    display_model_performance(result)

    # Step 9: Diagnostics
    n = int(result.nobs)
    diag = run_diagnostics(result, clean_df, x_col, y_col)
    display_diagnostics(diag, n)

    # Step 10: Prediction
    display_prediction(result, clean_df, x_col, y_col)

    # Step 11: Interpretation
    display_interpretation(result, diag, desc, x_col, y_col)

    # Step 12: Save
    save_report()

    rprint(f"\n{SEPARATOR}")
    rprint(f"  VM-WIT-STATS | Simple Linear Regression — Analysis Complete")
    rprint(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_simple_linear_regression()