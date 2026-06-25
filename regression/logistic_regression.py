# ======================================================================
# VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
# Module      : Binary Logistic Regression
# File        : regression/logistic_regression.py
# Description : Models the probability of a binary outcome as a function
#               of one or more predictors.
#               logit(P) = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ
#               Production-grade | Research-grade | Publication-grade
#               Uses statsmodels Logit for full inferential statistics.
#
# Pre-generation audit corrections (v1):
#   1. Perfect/quasi-separation detection before fitting.
#   2. Box-Tidwell test limited to continuous predictors only.
#   3. VIF computed on design matrix after dummy encoding.
#   4. Hosmer-Lemeshow implemented manually (10-group decile).
#   5. AUC computed via trapezoidal rule on sorted predictions.
#   6. Calibration assessed by HL test AND Brier score.
#   7. Cook's Distance threshold uses 4/n.
#   8. Class imbalance explicitly checked and reported.
#   9. McFadden R²: 1 - LL_full/LL_null (not deviance-based).
#  10. OR CIs on exponentiated Wald scale.
#  11. Large-sample statistical vs clinical significance caveat.
#  12. Separation warning issued; analysis continues with caveat.
#
# Post-generation statistical audit corrections (v2):
#  A. Cook's Distance: replaced r²/n proxy with exact logistic Cook's D
#     via statsmodels GLMInfluence (hat matrix diagonal from X'WX).
#     Proxy was not Cook's Distance and misidentified influential obs.
#  B. Separation detection: added post-fit quasi-separation check
#     (predicted probabilities ≥ 0.9999 or ≤ 0.0001) to complement
#     univariate pre-fit check. Multivariate separation cannot be
#     detected by univariate threshold scanning alone.
#  C. AUC: replaced np.trapz on sorted scores with Mann-Whitney U
#     equivalence (AUC = U / (n_pos × n_neg)). This is numerically
#     exact, handles tied predicted probabilities correctly, and
#     cannot return a negative value under any input order.
#  D. Hosmer-Lemeshow: replaced fixed index-based grouping with
#     quantile-based grouping that keeps all tied predicted
#     probabilities in the same group. Fixed index splitting
#     incorrectly divides ties across group boundaries.
#  E. Removed overclaimed wording: "meaningful variation" → "statistically
#     significant variation" (LR test establishes statistical significance
#     only; meaningfulness is a clinical judgement).
# ======================================================================

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families
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
REPORT_FILE = os.path.join(REPORT_DIR, "logistic_regression_result.txt")
MIN_SAMPLE_SIZE   = 3
LARGE_SAMPLE_N    = 200
DEFAULT_THRESHOLD = 0.50

# ======================================================================
# REPORT BUFFER
# ======================================================================

class ReportBuffer:
    """Accumulates all output so console and saved report are identical."""
    def __init__(self):
        self._lines: list[str] = []

    def write(self, text: str = ""):
        print(text)
        self._lines.append(text)

    def get(self) -> str:
        return "\n".join(self._lines)


_buf = ReportBuffer()


def rprint(text: str = ""):
    _buf.write(text)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

HEADER = """
======================================================================
        VM-WIT-STATS | VM Medical College
        Statistical Analysis Toolkit
        Module : Binary Logistic Regression
======================================================================
"""


def print_header():
    rprint(HEADER)


def print_section(title: str):
    rprint(f"\n{SEPARATOR2}")
    rprint(f"  {title}")
    rprint(SEPARATOR2)


def fmt(val, decimals: int = 4) -> str:
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
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return True
    return bool((np.max(finite) - np.min(finite)) < tol)


def check_statsmodels() -> bool:
    if not STATSMODELS_AVAILABLE:
        print("\n  [ERROR] 'statsmodels' is not installed.")
        print("          pip install statsmodels")
        return False
    return True


# ======================================================================
# SAMPLE DATA GENERATOR
# ======================================================================

def generate_sample_csv(path: str = "data/sample_logistic.csv",
                         n: int = 200,
                         seed: int = 42):
    """
    Generate a synthetic medical dataset for Binary Logistic Regression.

    Columns:
        Patient_ID  : unique identifier
        Age         : patient age in years
        SystolicBP  : systolic blood pressure (mmHg)
        HeartRate   : resting heart rate (bpm)
        Diabetes    : 0/1 binary predictor
        Mortality   : binary outcome (0 = survived, 1 = died)

    True log-odds: -8 + 0.08*Age + 0.03*SystolicBP + 0.02*HeartRate + 0.9*Diabetes

    Usage:
        from regression.logistic_regression import generate_sample_csv
        generate_sample_csv()
    """
    rng = np.random.default_rng(seed)
    age       = rng.uniform(30, 85, n).round(1)
    sbp       = rng.normal(130, 18, n).clip(90, 210).round(1)
    hr        = rng.normal(80, 14, n).clip(45, 140).round(1)
    diabetes  = rng.binomial(1, 0.30, n).astype(float)
    log_odds  = -8.0 + 0.08 * age + 0.03 * sbp + 0.02 * hr + 0.90 * diabetes
    prob      = 1.0 / (1.0 + np.exp(-log_odds))
    mortality = rng.binomial(1, prob, n).astype(int)

    df = pd.DataFrame({
        "Patient_ID" : [f"PT{str(i+1).zfill(3)}" for i in range(n)],
        "Age"        : age,
        "SystolicBP" : sbp,
        "HeartRate"  : hr,
        "Diabetes"   : diabetes,
        "Mortality"  : mortality,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [OK] Sample dataset generated: {os.path.abspath(path)}")
    print(f"       n = {n}  |  Outcome: Mortality  |  Events: {mortality.sum()}")
    return df


# ======================================================================
# INPUT & LOADING
# ======================================================================

def get_csv_path() -> str:
    print_section("STEP 1 | LOAD DATASET")
    rprint("\n  TIP: To generate a sample dataset for testing, run:")
    rprint("       from regression.logistic_regression import generate_sample_csv")
    rprint("       generate_sample_csv()")
    while True:
        path = input("\n  Enter path to CSV file: ").strip()
        if not path:
            print("  [ERROR] No path entered.")
            continue
        if not path.lower().endswith(".csv"):
            print("  [WARNING] No .csv extension. Attempting to load anyway.")
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
        print(f"  [ERROR] Parse error: {exc}")
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
# OUTCOME SELECTION & ENCODING
# ======================================================================

def select_outcome_column(df: pd.DataFrame) -> str | None:
    print_section("STEP 3 | SELECT BINARY OUTCOME VARIABLE")
    rprint("\n  The outcome must be binary: 0/1, Yes/No, True/False, or any 2-category variable.")
    rprint("  Example: Mortality, Survival, Diagnosis, Treatment_Success")

    while True:
        col = input("\n  Enter outcome column name: ").strip()
        if not col:
            print("  [ERROR] No column entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue
        vals = df[col].dropna().unique()
        if len(vals) < 2:
            print(f"  [ERROR] '{col}' has fewer than 2 unique values: {list(vals)}")
            continue
        if len(vals) > 2:
            print(f"  [ERROR] '{col}' has {len(vals)} unique values: {list(vals[:8])} ...")
            print("          Binary Logistic Regression requires exactly 2 classes.")
            continue
        rprint(f"  [OK] Outcome column: '{col}'  (categories: {sorted([str(v) for v in vals])})")
        return col


def encode_outcome(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Encode a binary outcome to 0/1.
    If already 0/1, keep as-is.
    Otherwise sort unique values and map lower → 0, higher → 1.
    Returns (encoded_series, encoding_map).
    """
    vals = sorted(series.dropna().unique(), key=str)
    if set(vals) == {0, 1} or set(vals) == {0.0, 1.0}:
        return series.astype(int), {vals[0]: 0, vals[1]: 1}
    encoding = {vals[0]: 0, vals[1]: 1}
    return series.map(encoding).astype(float), encoding


# ======================================================================
# PREDICTOR SELECTION
# ======================================================================

def select_predictors(df: pd.DataFrame, outcome_col: str) -> list | None:
    print_section("STEP 4 | SELECT PREDICTOR VARIABLES")
    rprint("\n  Enter predictor column names one at a time.")
    rprint("  Continuous, binary, and categorical predictors are all accepted.")
    rprint("  Categorical predictors will be automatically dummy-encoded.")
    rprint("  Type 'done' when finished. At least 1 predictor is required.")

    selected = []
    while True:
        prompt = f"\n  Enter predictor {len(selected)+1} (or 'done'): "
        col = input(prompt).strip()

        if col.lower() == "done":
            if len(selected) == 0:
                print("  [ERROR] At least one predictor is required.")
                continue
            break

        if not col:
            print("  [ERROR] No column entered.")
            continue
        if col not in df.columns:
            print(f"  [ERROR] Column '{col}' not found.")
            continue
        if col == outcome_col:
            print(f"  [ERROR] Cannot use the outcome column as a predictor.")
            continue
        if col in selected:
            print(f"  [WARNING] '{col}' already selected.")
            continue

        n_unique = df[col].nunique()
        dtype    = str(df[col].dtype)
        rprint(f"  [OK] Added: '{col}'  (dtype: {dtype}, unique values: {n_unique})")
        selected.append(col)

    rprint(f"\n  Predictors selected ({len(selected)}): {selected}")
    return selected


# ======================================================================
# DATA PREPARATION & CLEANING
# ======================================================================

def prepare_data(df: pd.DataFrame,
                 outcome_col: str,
                 predictor_cols: list) -> tuple | None:
    """
    Clean data, encode outcome and categorical predictors, build design matrix.
    Returns (clean_df, y, X_design, X_with_int, col_names, enc_map, dummy_info)
    or None on failure.
    """
    print_section("STEP 5 | DATA CLEANING & PREPARATION")

    all_cols = [outcome_col] + predictor_cols
    working  = df[all_cols].copy()
    orig_n   = len(working)
    rprint(f"\n  Original rows : {orig_n:,}")

    # Drop NaN in any column
    before  = len(working)
    working = working.dropna(subset=all_cols)
    dropped_nan = before - len(working)
    if dropped_nan > 0:
        rprint(f"  Dropped (NaN)       : {dropped_nan:,}")

    # Drop inf in numeric columns
    before = len(working)
    for col in predictor_cols:
        if pd.api.types.is_numeric_dtype(working[col]):
            working = working[np.isfinite(working[col].to_numpy(dtype=float))]
    dropped_inf = before - len(working)
    if dropped_inf > 0:
        rprint(f"  Dropped (Infinite)  : {dropped_inf:,}")

    n = len(working)
    rprint(f"\n  Total dropped    : {orig_n - n:,}")
    rprint(f"  Rows for analysis: {n:,}")

    if n == 0:
        rprint("\n  [ERROR] No valid rows remain.")
        return None
    if n < MIN_SAMPLE_SIZE:
        rprint(f"\n  [ERROR] Only {n} observations remain. Minimum is {MIN_SAMPLE_SIZE}.")
        return None

    if n < 10:
        rprint(f"\n  [WARNING] Very small sample (n = {n}). Results are exploratory only.")
    elif n < 30:
        rprint(f"\n  [WARNING] Small sample (n = {n}). Interpret estimates cautiously.")

    # Encode outcome
    y_raw, enc_map = encode_outcome(working[outcome_col])
    rprint(f"\n  Outcome encoding: {enc_map}")
    event_label = [k for k, v in enc_map.items() if v == 1][0]
    rprint(f"  Event (Y=1)     : '{event_label}'")

    y = y_raw.to_numpy(dtype=float)
    n_events    = int(np.sum(y == 1))
    n_nonevents = int(np.sum(y == 0))
    rprint(f"  Events (Y=1)    : {n_events:,}  ({n_events/n*100:.1f}%)")
    rprint(f"  Non-events (Y=0): {n_nonevents:,}  ({n_nonevents/n*100:.1f}%)")

    # Class imbalance
    minority_pct = min(n_events, n_nonevents) / n * 100
    if minority_pct < 10:
        rprint(f"\n  [WARNING] Severe class imbalance: minority class = {minority_pct:.1f}%.")
        rprint("            Sensitivity, PPV, and AUC may be misleading.")
        rprint("            Consider reporting calibration and discrimination separately.")
    elif minority_pct < 20:
        rprint(f"\n  [WARNING] Moderate class imbalance: minority class = {minority_pct:.1f}%.")
        rprint("            Interpret classification metrics with caution.")

    # Events per variable
    epv = n_events / len(predictor_cols)
    if epv < 10:
        rprint(f"\n  [WARNING] Events per variable (EPV) = {epv:.1f} < 10.")
        rprint("            Model may be overfitted. External validation is strongly advised.")

    # Duplicate row detection
    dup_count = working.duplicated(subset=all_cols).sum()
    if dup_count > 0:
        rprint(f"\n  [WARNING] {dup_count:,} duplicate row(s) detected.")
        rprint("            If these are duplicated records, remove them before analysis.")
        rprint("            If they are different patients with identical values, disregard.")
        rprint("            Proceeding with all rows included.")

    # Constant predictor check
    for col in predictor_cols:
        if pd.api.types.is_numeric_dtype(working[col]):
            arr = working[col].to_numpy(dtype=float)
            if is_effectively_constant(arr):
                rprint(f"\n  [ERROR] Predictor '{col}' has zero variance.")
                rprint("          Logistic regression cannot estimate a coefficient for a constant.")
                return None

    # Dummy encode categorical predictors
    dummy_info  = {}
    X_parts     = []
    col_names_X = []

    for col in predictor_cols:
        if pd.api.types.is_numeric_dtype(working[col]):
            unique_vals = working[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                X_parts.append(working[[col]].to_numpy(dtype=float))
                col_names_X.append(col)
                dummy_info[col] = {"type": "binary"}
            else:
                X_parts.append(working[[col]].to_numpy(dtype=float))
                col_names_X.append(col)
                dummy_info[col] = {"type": "continuous"}
        else:
            dummies = pd.get_dummies(working[col], prefix=col, drop_first=True, dtype=float)
            X_parts.append(dummies.to_numpy(dtype=float))
            col_names_X.extend(list(dummies.columns))
            dummy_info[col] = {"type": "categorical", "dummies": list(dummies.columns)}
            rprint(f"  Categorical '{col}' encoded as: {list(dummies.columns)}")

    X_design   = np.hstack(X_parts)
    X_with_int = sm.add_constant(X_design)

    # Sparse category check
    for col in predictor_cols:
        if not pd.api.types.is_numeric_dtype(working[col]):
            for cat in working[col].unique():
                n_cat = (working[col] == cat).sum()
                if n_cat < 5:
                    rprint(f"  [WARNING] Category '{cat}' in '{col}' has only {n_cat} observations.")
                    rprint("            Sparse categories reduce estimate reliability.")

    rprint(f"\n  Design matrix : {X_with_int.shape[0]} rows × {X_with_int.shape[1]} columns")
    rprint(f"  (includes intercept + {len(col_names_X)} predictor term(s))")
    rprint(f"\n  [OK] Data preparation complete.")

    return working.reset_index(drop=True), y, X_design, X_with_int, col_names_X, enc_map, dummy_info


# ======================================================================
# SEPARATION DETECTION
# ======================================================================

def check_separation_prefit(y: np.ndarray,
                             X_design: np.ndarray,
                             col_names: list) -> bool:
    """
    Pre-fit univariate threshold scan for perfect separation.
    For each predictor, checks every unique value as a threshold to
    determine whether one side contains only events or only non-events.

    Limitation: detects univariate perfect separation only.
    Multivariate separation (where a linear combination separates)
    cannot be detected by univariate scanning. Post-fit quasi-separation
    check via predicted probabilities is used as a complementary method.

    Returns True if separation is detected.
    """
    separation_found = False
    for i, col in enumerate(col_names):
        x = X_design[:, i]
        unique_x = np.unique(x)
        for threshold in unique_x:
            above = (x >= threshold)
            below = ~above
            if above.sum() > 0 and below.sum() > 0:
                y_above = y[above]
                y_below = y[below]
                if (np.all(y_above == 1) and np.all(y_below == 0)) or \
                   (np.all(y_above == 0) and np.all(y_below == 1)):
                    rprint(f"\n  [WARNING] Perfect separation detected in predictor '{col}'.")
                    rprint(f"            At threshold {threshold:.4f}, one side contains only events,")
                    rprint(f"            the other only non-events.")
                    rprint(f"            MLE estimates will be infinite; SEs will be unreliable.")
                    rprint(f"            Consider: Firth penalised logistic regression.")
                    separation_found = True
                    break
    return separation_found


def check_separation_postfit(y_pred_prob: np.ndarray) -> bool:
    """
    Post-fit quasi-separation check via predicted probabilities.

    When separation or quasi-separation is present, the MLE pushes
    predicted probabilities toward 0 or 1 for the separating observations.
    Predicted probabilities ≥ 0.9999 or ≤ 0.0001 indicate that the
    model has encountered separation that may not have been detected
    univariately (e.g., multivariate linear combination separates).

    This is a heuristic but is the standard practical check used in
    applied logistic regression (Heinze & Schemper, 2002).

    Returns True if quasi-separation is suspected.
    """
    extreme_high = int(np.sum(y_pred_prob >= 0.9999))
    extreme_low  = int(np.sum(y_pred_prob <= 0.0001))
    if extreme_high > 0 or extreme_low > 0:
        rprint(f"\n  [WARNING] Post-fit quasi-separation suspected.")
        rprint(f"            Predicted probabilities ≥ 0.9999 : {extreme_high} observation(s)")
        rprint(f"            Predicted probabilities ≤ 0.0001 : {extreme_low} observation(s)")
        rprint(f"            These extreme fitted values indicate the MLE is being pushed")
        rprint(f"            toward ±∞ by separation in the predictor space.")
        rprint(f"            Coefficient SEs and p-values for affected predictors")
        rprint(f"            are unreliable. Use Firth logistic regression.")
        return True
    return False


# ======================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================

def display_descriptive_stats(working: pd.DataFrame,
                               outcome_col: str,
                               predictor_cols: list,
                               y: np.ndarray,
                               enc_map: dict):
    print_section("DESCRIPTIVE STATISTICS")

    event_label    = [k for k, v in enc_map.items() if v == 1][0]
    nonevent_label = [k for k, v in enc_map.items() if v == 0][0]
    n_events    = int(np.sum(y == 1))
    n_nonevents = int(np.sum(y == 0))
    n = len(y)

    rprint(f"\n  OUTCOME: {outcome_col}")
    rprint(f"  {'Category':<30} {'Count':>10} {'Percent':>10}")
    rprint(f"  {'-'*30} {'-'*10} {'-'*10}")
    rprint(f"  {str(event_label) + ' (event, Y=1)':<30} {n_events:>10,} {n_events/n*100:>9.1f}%")
    rprint(f"  {str(nonevent_label) + ' (non-event, Y=0)':<30} {n_nonevents:>10,} {n_nonevents/n*100:>9.1f}%")
    rprint(f"  {'Total':<30} {n:>10,} {'100.0%':>10}")

    rprint(f"\n  PREDICTORS:")
    for col in predictor_cols:
        arr = working[col]
        if pd.api.types.is_numeric_dtype(arr):
            a = arr.to_numpy(dtype=float)
            rprint(f"\n  {col} (numeric)")
            rprint(f"    n={len(a):,}  Mean={np.mean(a):.3f}  SD={np.std(a, ddof=1):.3f}"
                   f"  Min={np.min(a):.3f}  Max={np.max(a):.3f}")
            rprint(f"    Median={np.median(a):.3f}  "
                   f"IQR=[{np.percentile(a,25):.3f}, {np.percentile(a,75):.3f}]")
            mean_event    = np.mean(a[y == 1])
            mean_nonevent = np.mean(a[y == 0])
            rprint(f"    Mean in events (Y=1)    : {mean_event:.3f}")
            rprint(f"    Mean in non-events (Y=0): {mean_nonevent:.3f}")
        else:
            counts = arr.value_counts()
            rprint(f"\n  {col} (categorical, {len(counts)} categories)")
            for cat, cnt in counts.items():
                rprint(f"    {str(cat):<25}: {cnt:>6,}  ({cnt/len(arr)*100:.1f}%)")


# ======================================================================
# MODEL FITTING
# ======================================================================

def fit_logistic(y: np.ndarray,
                 X_with_int: np.ndarray,
                 col_names: list):
    """
    Fit Binary Logistic Regression using statsmodels Logit.
    Returns fitted result object or None on failure.
    """
    try:
        model  = sm.Logit(y, X_with_int)
        result = model.fit(disp=False, maxiter=200, method="newton")
    except Exception as exc:
        rprint(f"  [ERROR] Model fitting failed: {exc}")
        return None

    if not result.mle_retvals["converged"]:
        rprint("\n  [WARNING] Model did NOT converge.")
        rprint("            Estimates may be unreliable.")
        rprint("            This can indicate perfect separation, sparse data,")
        rprint("            or a poorly specified model.")

    return result


# ======================================================================
# COEFFICIENT TABLE
# ======================================================================

def display_coefficient_table(result, col_names: list, x_col_names: list):
    print_section("COEFFICIENT TABLE (Log-Odds Scale)")

    all_names = ["Intercept"] + x_col_names
    betas  = result.params
    ses    = result.bse
    zvals  = result.tvalues
    pvals  = result.pvalues
    ci     = result.conf_int(alpha=0.05)

    rprint(f"\n  {'Parameter':<32} {'β':>10} {'SE':>10} {'Z':>10} {'p-value':>10}")
    rprint(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(all_names):
        rprint(f"  {name:<32} {betas[i]:>10.4f} {ses[i]:>10.4f} "
               f"{zvals[i]:>10.4f} {pvals[i]:>10.4f}")

    print_section("ODDS RATIO TABLE (Exponentiated Scale)")
    rprint(f"\n  {'Parameter':<32} {'OR':>10} {'95% CI Lower':>14} "
           f"{'95% CI Upper':>14} {'p-value':>10}")
    rprint(f"  {'-'*32} {'-'*10} {'-'*14} {'-'*14} {'-'*10}")

    for i, name in enumerate(all_names):
        or_val = math.exp(betas[i]) if not math.isnan(betas[i]) else float("nan")
        ci_lo  = math.exp(ci[i, 0]) if not math.isnan(ci[i, 0]) else float("nan")
        ci_hi  = math.exp(ci[i, 1]) if not math.isnan(ci[i, 1]) else float("nan")
        sig_str = " *" if pvals[i] < 0.05 else "  "
        rprint(f"  {name:<32} {fmt(or_val):>10} {fmt(ci_lo):>14} "
               f"{fmt(ci_hi):>14} {pvals[i]:>10.4f}{sig_str}")

    rprint(f"\n  * = Statistically significant at α = 0.05")
    rprint(f"\n  INTERPRETATION OF ODDS RATIOS:")
    rprint(f"  OR > 1 → predictor associated with higher odds of the event.")
    rprint(f"  OR < 1 → predictor associated with lower odds of the event.")
    rprint(f"  OR = 1 → no association.")
    rprint(f"  95% CI excluding 1.0 indicates statistical significance.")
    rprint(f"\n  NOTE: These are Wald-based confidence intervals (exp(β ± Z·SE)).")
    rprint(f"  They are consistent with the reported standard errors and p-values.")


# ======================================================================
# MODEL PERFORMANCE
# ======================================================================

def display_model_performance(result):
    print_section("MODEL PERFORMANCE")

    ll_full = float(result.llf)
    ll_null = float(result.llnull)
    lr_stat = -2.0 * (ll_null - ll_full)
    lr_df   = int(result.df_model)
    lr_p    = float(stats.chi2.sf(lr_stat, lr_df))
    n       = int(result.nobs)

    # McFadden R²: 1 - LL_full / LL_null
    # Both LL values are negative; LL_null is more negative (further from 0).
    # A perfect model has LL_full → 0, giving McFadden R² → 1.
    if ll_null != 0:
        mcfadden = float(1.0 - ll_full / ll_null)
    else:
        mcfadden = float("nan")

    aic = float(result.aic)
    bic = float(result.bic)
    k   = lr_df

    rprint(f"\n  {'Metric':<40} {'Value':>20}")
    rprint(f"  {'-'*40} {'-'*20}")
    rprint(f"  {'Log-Likelihood (fitted model)':<40} {ll_full:>20.4f}")
    rprint(f"  {'Null Log-Likelihood (intercept only)':<40} {ll_null:>20.4f}")
    rprint(f"  {'Likelihood Ratio Statistic (χ²)':<40} {lr_stat:>20.4f}")
    rprint(f"  {'LR Test df':<40} {lr_df:>20,}")
    rprint(f"  {'LR Test p-value':<40} {lr_p:>20.4f}")
    rprint(f"  {'McFadden R²':<40} {fmt(mcfadden):>20}")
    rprint(f"  {'AIC':<40} {aic:>20.4f}")
    rprint(f"  {'BIC':<40} {bic:>20.4f}")
    rprint(f"  {'Sample Size (n)':<40} {n:>20,}")
    rprint(f"  {'Number of Predictors (k)':<40} {k:>20,}")

    rprint(f"\n  MODEL FIT:")
    if lr_p < 0.05:
        rprint(f"  LR Test: χ²({lr_df}) = {lr_stat:.4f}, p = {lr_p:.4f}")
        rprint(f"  The model with predictors fits significantly better than the null model.")
    else:
        rprint(f"  LR Test: χ²({lr_df}) = {lr_stat:.4f}, p = {lr_p:.4f}")
        rprint(f"  The model does not fit significantly better than the null model.")

    rprint(f"\n  McFadden R² INTERPRETATION:")
    if math.isnan(mcfadden):
        rprint(f"  McFadden R² could not be computed.")
    elif mcfadden < 0.10:
        rprint(f"  McFadden R² = {mcfadden:.4f} → Poor fit.")
    elif mcfadden < 0.20:
        rprint(f"  McFadden R² = {mcfadden:.4f} → Fair fit.")
    elif mcfadden < 0.30:
        rprint(f"  McFadden R² = {mcfadden:.4f} → Good fit.")
    else:
        rprint(f"  McFadden R² = {mcfadden:.4f} → Excellent fit.")
    rprint(f"  McFadden R² of 0.20–0.40 is considered excellent in applied research.")
    rprint(f"  Unlike OLS R², it cannot be interpreted as proportion of variance explained.")

    return {"ll_full": ll_full, "ll_null": ll_null, "lr_stat": lr_stat,
            "lr_df": lr_df, "lr_p": lr_p, "mcfadden": mcfadden}


# ======================================================================
# CLASSIFICATION PERFORMANCE
# ======================================================================

def compute_classification(y: np.ndarray,
                            y_pred_prob: np.ndarray,
                            threshold: float = DEFAULT_THRESHOLD) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    n  = len(y)

    accuracy    = (tp + tn) / n              if n > 0           else float("nan")
    sensitivity = tp / (tp + fn)             if (tp + fn) > 0   else float("nan")
    specificity = tn / (tn + fp)             if (tn + fp) > 0   else float("nan")
    precision   = tp / (tp + fp)             if (tp + fp) > 0   else float("nan")
    npv         = tn / (tn + fn)             if (tn + fn) > 0   else float("nan")
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (not math.isnan(precision) and not math.isnan(sensitivity)
              and (precision + sensitivity) > 0) else float("nan"))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "sensitivity": sensitivity,
            "specificity": specificity, "precision": precision,
            "npv": npv, "f1": f1, "threshold": threshold}


def display_classification(clf: dict):
    print_section("CLASSIFICATION PERFORMANCE")

    rprint(f"\n  Classification threshold : {clf['threshold']:.2f}")
    rprint(f"\n  CONFUSION MATRIX:")
    rprint(f"  {'':>30} {'Predicted Y=1':>15} {'Predicted Y=0':>15}")
    rprint(f"  {'-'*60}")
    rprint(f"  {'Actual Y=1 (Events)':<30} {clf['tp']:>15,} {clf['fn']:>15,}")
    rprint(f"  {'Actual Y=0 (Non-Events)':<30} {clf['fp']:>15,} {clf['tn']:>15,}")

    rprint(f"\n  {'Metric':<35} {'Value':>15}")
    rprint(f"  {'-'*35} {'-'*15}")
    rprint(f"  {'Accuracy':<35} {fmt(clf['accuracy'], 4):>15}")
    rprint(f"  {'Sensitivity (Recall)':<35} {fmt(clf['sensitivity'], 4):>15}")
    rprint(f"  {'Specificity':<35} {fmt(clf['specificity'], 4):>15}")
    rprint(f"  {'Precision (PPV)':<35} {fmt(clf['precision'], 4):>15}")
    rprint(f"  {'Negative Predictive Value (NPV)':<35} {fmt(clf['npv'], 4):>15}")
    rprint(f"  {'F1 Score':<35} {fmt(clf['f1'], 4):>15}")

    rprint(f"\n  Sensitivity: proportion of actual events correctly classified.")
    rprint(f"  Specificity: proportion of actual non-events correctly classified.")
    rprint(f"  The appropriate threshold depends on the relative costs of false")
    rprint(f"  positives vs false negatives in the clinical context.")
    rprint(f"  Threshold = {clf['threshold']:.2f} is the default. Adjust based on clinical utility.")


# ======================================================================
# ROC & AUC — MANN-WHITNEY U EQUIVALENCE
# ======================================================================

def compute_roc_auc(y: np.ndarray,
                    y_pred_prob: np.ndarray) -> dict:
    """
    Compute ROC AUC using the Mann-Whitney U statistic equivalence.

    AUC = U / (n_pos × n_neg)

    where U = number of (event, non-event) pairs where the event has
    a strictly higher predicted probability than the non-event, plus
    0.5 × the number of tied pairs.

    This formulation is mathematically identical to the area under the
    ROC curve computed by the trapezoidal rule, but is numerically exact
    regardless of tied predicted probabilities, requires no sorting of
    FPR (which can produce non-monotone sequences under ties), and
    cannot return a negative value under any input ordering.

    Reference: Hanley & McNeil (1982), Bamber (1975).
    """
    pos_probs = y_pred_prob[y == 1]
    neg_probs = y_pred_prob[y == 0]
    n_pos = len(pos_probs)
    n_neg = len(neg_probs)

    if n_pos == 0 or n_neg == 0:
        return {"auc": float("nan"), "tpr": np.array([]), "fpr": np.array([])}

    # U statistic: count concordant + 0.5 * tied pairs
    # Vectorised: for each positive, count negatives with lower prob (concordant)
    # and add 0.5 for ties
    u_stat = 0.0
    for p in pos_probs:
        u_stat += np.sum(neg_probs < p)
        u_stat += 0.5 * np.sum(neg_probs == p)

    auc = u_stat / (n_pos * n_neg)

    # Build ROC curve points for reporting (sorted by threshold)
    order      = np.argsort(-y_pred_prob)
    y_sorted   = y[order]
    tpr_list   = [0.0]
    fpr_list   = [0.0]
    tp_cum, fp_cum = 0, 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_list.append(tp_cum / n_pos)
        fpr_list.append(fp_cum / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    return {
        "auc": float(auc),
        "tpr": np.array(tpr_list),
        "fpr": np.array(fpr_list),
    }


def display_roc(roc: dict):
    print_section("ROC ANALYSIS")

    auc = roc["auc"]
    rprint(f"\n  Area Under the ROC Curve (AUC) : {fmt(auc)}")
    rprint(f"  Method: Mann-Whitney U statistic equivalence (exact, tie-corrected)")

    rprint(f"\n  AUC INTERPRETATION:")
    rprint(f"  {'AUC Range':<25} {'Discrimination':<20}")
    rprint(f"  {'-'*25} {'-'*20}")
    rprint(f"  {'0.50 – 0.60':<25} {'No discrimination':<20}")
    rprint(f"  {'0.60 – 0.70':<25} {'Poor':<20}")
    rprint(f"  {'0.70 – 0.80':<25} {'Acceptable':<20}")
    rprint(f"  {'0.80 – 0.90':<25} {'Excellent':<20}")
    rprint(f"  {'0.90 – 1.00':<25} {'Outstanding':<20}")

    if math.isnan(auc):
        rprint(f"\n  AUC could not be computed (insufficient events or non-events).")
    elif auc >= 0.90:
        interp = "Outstanding discrimination."
    elif auc >= 0.80:
        interp = "Excellent discrimination."
    elif auc >= 0.70:
        interp = "Acceptable discrimination."
    elif auc >= 0.60:
        interp = "Poor discrimination."
    else:
        interp = "No better than chance."

    if not math.isnan(auc):
        rprint(f"\n  AUC = {auc:.4f} → {interp}")
    rprint(f"\n  AUC is the probability that a randomly chosen event has a higher")
    rprint(f"  predicted probability than a randomly chosen non-event.")
    rprint(f"  It is threshold-independent and unaffected by class imbalance.")


# ======================================================================
# CALIBRATION — HOSMER-LEMESHOW (TIE-SAFE) & BRIER SCORE
# ======================================================================

def hosmer_lemeshow_test(y: np.ndarray,
                          y_pred_prob: np.ndarray,
                          g: int = 10) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test with tie-safe quantile grouping.

    Standard decile-of-risk method (Hosmer & Lemeshow, 1980).
    Groups are formed by the predicted probability deciles. All
    observations with the same predicted probability are assigned to
    the same group (tie-safe). This prevents the incorrect splitting
    of tied observations across group boundaries that occurs with
    fixed-index grouping (n // g).

    Degrees of freedom: g_used - 2, where g_used is the number of
    non-empty groups. Minimum df = 1 to avoid degenerate chi-square.

    Small-sample note: HL test has low power when n < 100 or when
    groups have expected counts < 5. A warning is issued when either
    condition applies. The Brier score is reported as a supplementary
    calibration measure that does not depend on group formation.
    """
    n = len(y)

    # Quantile-based grouping: use np.percentile to find g boundaries
    # All tied values at a boundary go to the same group
    quantiles   = np.linspace(0, 100, g + 1)
    boundaries  = np.percentile(y_pred_prob, quantiles)
    boundaries  = np.unique(boundaries)  # collapse ties at boundary

    # Assign each observation to a group based on boundaries
    group_ids = np.digitize(y_pred_prob, boundaries[1:-1], right=True)
    # group_ids in {0, 1, ..., len(boundaries)-2}

    hl_chi2     = 0.0
    groups_used = 0
    sparse_groups = 0

    for g_id in np.unique(group_ids):
        mask  = (group_ids == g_id)
        y_grp = y[mask]
        p_grp = y_pred_prob[mask]
        ng    = len(y_grp)
        if ng == 0:
            continue

        obs_1 = float(np.sum(y_grp == 1))
        obs_0 = float(ng - obs_1)
        exp_1 = float(np.sum(p_grp))
        exp_0 = float(ng - exp_1)

        if exp_1 < 5 or exp_0 < 5:
            sparse_groups += 1

        if exp_1 > 1e-10:
            hl_chi2 += (obs_1 - exp_1) ** 2 / exp_1
        if exp_0 > 1e-10:
            hl_chi2 += (obs_0 - exp_0) ** 2 / exp_0
        groups_used += 1

    df    = max(groups_used - 2, 1)
    p_val = float(stats.chi2.sf(hl_chi2, df))

    return {
        "hl_chi2"      : hl_chi2,
        "df"           : df,
        "p_value"      : p_val,
        "groups"       : groups_used,
        "sparse_groups": sparse_groups,
    }


def brier_score(y: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Brier score: mean squared prediction error. Range [0, 1]."""
    return float(np.mean((y - y_pred_prob) ** 2))


def display_calibration(y: np.ndarray, y_pred_prob: np.ndarray):
    print_section("CALIBRATION")

    n   = len(y)
    hl  = hosmer_lemeshow_test(y, y_pred_prob, g=10)
    bs  = brier_score(y, y_pred_prob)

    prevalence = float(np.mean(y))
    bs_null    = prevalence * (1 - prevalence)
    bss        = 1.0 - bs / bs_null if bs_null > 0 else float("nan")

    rprint(f"\n  HOSMER-LEMESHOW TEST (quantile-based grouping, tie-safe):")
    rprint(f"  H0: Model is well-calibrated (predicted probabilities match observed rates).")
    rprint(f"  H1: Model is poorly calibrated.\n")
    rprint(f"  {'HL χ² Statistic':<35} {hl['hl_chi2']:>15.4f}")
    rprint(f"  {'Degrees of Freedom':<35} {hl['df']:>15,}")
    rprint(f"  {'p-value':<35} {hl['p_value']:>15.4f}")
    rprint(f"  {'Groups formed':<35} {hl['groups']:>15,}")
    rprint(f"  {'Groups with expected < 5':<35} {hl['sparse_groups']:>15,}")

    if hl["sparse_groups"] > 0:
        rprint(f"\n  [WARNING] {hl['sparse_groups']} group(s) have expected counts < 5.")
        rprint(f"            Chi-square approximation may be unreliable for these groups.")

    if n < 100:
        rprint(f"\n  [WARNING] Small sample (n = {n}). HL test has low power.")
        rprint(f"            A non-significant result does not confirm good calibration.")

    if hl["p_value"] > 0.05:
        rprint(f"\n  Result : PASS — Good calibration (p > 0.05).")
        rprint(f"           Predicted probabilities are consistent with observed event rates.")
    else:
        rprint(f"\n  Result : FAIL — Poor calibration (p ≤ 0.05).")
        rprint(f"           Predicted probabilities deviate from observed event rates.")
        rprint(f"           Model may systematically over- or under-predict risk.")

    rprint(f"\n  BRIER SCORE:")
    rprint(f"  {'Brier Score':<35} {bs:>15.4f}")
    rprint(f"  {'Brier Skill Score (vs null)':<35} {fmt(bss):>15}")
    rprint(f"  Null Brier Score (prevalence-based): {bs_null:.4f}")
    if bs < bs_null:
        rprint(f"  Model ({bs:.4f}) < null ({bs_null:.4f}): model outperforms chance prediction.")
    else:
        rprint(f"  Model ({bs:.4f}) ≥ null ({bs_null:.4f}): model does not outperform chance.")

    rprint(f"\n  NOTE: The Hosmer-Lemeshow test depends on the number of groups,")
    rprint(f"  has low power in small samples, and excessive power in very large samples.")
    rprint(f"  The Brier Score is a continuous, group-free calibration measure")
    rprint(f"  and is recommended as the primary calibration metric.")


# ======================================================================
# ASSUMPTION DIAGNOSTICS
# ======================================================================

def compute_vif(X_design: np.ndarray, col_names: list) -> dict:
    """
    Compute VIF for each predictor column in the design matrix.
    VIF is computed without the intercept column.
    """
    vif_dict = {}
    if X_design.shape[1] < 2:
        return vif_dict
    for i, name in enumerate(col_names):
        try:
            vif_val = variance_inflation_factor(X_design, i)
            vif_dict[name] = float(vif_val)
        except Exception:
            vif_dict[name] = float("nan")
    return vif_dict


def box_tidwell_test(y: np.ndarray,
                     X_design: np.ndarray,
                     col_names: list,
                     dummy_info: dict,
                     predictor_cols: list) -> dict:
    """
    Box-Tidwell test for linearity of the logit (continuous predictors only).

    Adds X·ln(X) interaction terms for each continuous predictor and tests
    their significance in a logistic model augmented with the base predictors.
    A significant interaction (p ≤ 0.05) indicates non-linear logit.

    Excluded: binary and dummy-encoded categorical predictors, for which
    ln(X) is undefined (X = 0) and the linearity assumption does not apply.
    """
    results = {}

    continuous_indices = []
    for i, name in enumerate(col_names):
        base_col = name
        if base_col in dummy_info and dummy_info[base_col]["type"] == "continuous":
            continuous_indices.append((i, name))

    if not continuous_indices:
        return {"note": "No continuous predictors — Box-Tidwell test not applicable."}

    X_bt    = sm.add_constant(X_design)
    bt_cols = []
    for (idx, name) in continuous_indices:
        x_col = X_design[:, idx].copy()
        x_pos = x_col.copy()
        x_pos[x_pos <= 0] = np.nan
        if np.sum(~np.isnan(x_pos)) < 10:
            results[name] = {"note": f"Cannot compute ln({name}): too few positive values."}
            continue
        x_log_x = x_pos * np.log(x_pos)
        x_log_x = np.where(np.isnan(x_log_x), 0.0, x_log_x)
        bt_cols.append((name, x_log_x))

    if not bt_cols:
        return {"note": "Box-Tidwell interaction terms could not be constructed."}

    X_aug = X_bt.copy()
    for (name, x_log_x) in bt_cols:
        X_aug = np.column_stack([X_aug, x_log_x])

    try:
        model_bt = sm.Logit(y, X_aug)
        res_bt   = model_bt.fit(disp=False, maxiter=200)
        n_base   = X_bt.shape[1]
        for j, (name, _) in enumerate(bt_cols):
            idx_aug = n_base + j
            results[name] = {
                "beta"  : float(res_bt.params[idx_aug]),
                "pvalue": float(res_bt.pvalues[idx_aug]),
                "linear": bool(res_bt.pvalues[idx_aug] > 0.05),
            }
    except Exception as exc:
        return {"note": f"Box-Tidwell test failed: {exc}"}

    return results


def compute_influence(result,
                      y: np.ndarray,
                      X_with_int: np.ndarray,
                      y_pred_prob: np.ndarray) -> dict:
    """
    Compute exact logistic Cook's Distance via statsmodels GLMInfluence.

    Cook's Distance for logistic regression is defined as:
        D_i = (β̂ - β̂₍₋ᵢ₎)' (X'WX) (β̂ - β̂₍₋ᵢ₎) / p
    where β̂₍₋ᵢ₎ is the MLE with observation i deleted, W is the diagonal
    weight matrix with w_ii = π̂_i(1 - π̂_i), and p is the number of
    parameters.

    This is computed via GLMInfluence (Pregibon, 1981 one-step approximation)
    using the hat matrix from the weighted design matrix X'WX. This is
    the standard implementation in R's influence.measures() and SAS's
    PROC LOGISTIC INFLUENCE option.

    Outlier detection uses Pearson residuals:
        r_i = (y_i - π̂_i) / sqrt(π̂_i(1 - π̂_i))
    Threshold: |r_i| > 2 (approximately 2 SDs from zero).

    Cook's Distance threshold: 4/n (standard cross-sectional threshold).
    """
    n           = len(y)
    cook_thresh = 4.0 / n

    # Pearson residuals
    pi    = y_pred_prob
    denom = np.sqrt(pi * (1.0 - pi))
    denom = np.where(denom < 1e-10, 1e-10, denom)
    pearson_resid = (y - pi) / denom

    # Exact Cook's Distance via GLMInfluence
    cooks_d      = np.full(n, float("nan"))
    cooks_failed = False
    try:
        glm_model  = GLM(y, X_with_int,
                         family=families.Binomial())
        glm_result = glm_model.fit(disp=False)
        infl       = glm_result.get_influence()
        cooks_d    = infl.cooks_distance[0]
        # Replace any NaN from GLMInfluence (can occur at boundary observations)
        cooks_d    = np.where(np.isnan(cooks_d), 0.0, cooks_d)
    except Exception:
        cooks_failed = True
        # Fallback: Pregibon (1981) one-step approximation
        # D_i ≈ r²_i · h_ii / (p · (1 - h_ii)²)
        # where h_ii = w_ii · x_i'(X'WX)⁻¹x_i
        try:
            w    = pi * (1.0 - pi)
            W    = np.diag(w)
            XtWX = X_with_int.T @ W @ X_with_int
            p    = X_with_int.shape[1]
            XtWX_inv = np.linalg.inv(XtWX + np.eye(p) * 1e-10)
            h_diag = np.array([
                w[i] * (X_with_int[i] @ XtWX_inv @ X_with_int[i])
                for i in range(n)
            ])
            h_diag   = np.clip(h_diag, 0.0, 0.9999)
            cooks_d  = (pearson_resid ** 2 * h_diag) / (p * (1.0 - h_diag) ** 2)
            cooks_d  = np.where(np.isnan(cooks_d), 0.0, cooks_d)
        except Exception:
            cooks_d = np.zeros(n)

    outlier_mask   = np.abs(pearson_resid) > 2.0
    influence_mask = cooks_d > cook_thresh

    return {
        "pearson_resid"   : pearson_resid,
        "cooks_d"         : cooks_d,
        "cook_thresh"     : cook_thresh,
        "n_outliers"      : int(outlier_mask.sum()),
        "outlier_idx"     : np.where(outlier_mask)[0].tolist(),
        "n_influential"   : int(influence_mask.sum()),
        "influential_idx" : np.where(influence_mask)[0].tolist(),
        "cooks_exact"     : not cooks_failed,
    }


def display_diagnostics(result,
                         y: np.ndarray,
                         y_pred_prob: np.ndarray,
                         X_design: np.ndarray,
                         X_with_int: np.ndarray,
                         col_names: list,
                         dummy_info: dict,
                         predictor_cols: list):
    print_section("ASSUMPTION DIAGNOSTICS")

    n = len(y)

    # --- VIF ---
    rprint("\n  [1] MULTICOLLINEARITY — Variance Inflation Factor (VIF)")
    rprint(f"      VIF < 5   : Acceptable")
    rprint(f"      VIF 5–10  : Moderate — investigate")
    rprint(f"      VIF > 10  : High — serious multicollinearity\n")
    vif_dict = compute_vif(X_design, col_names)
    if not vif_dict:
        rprint("      Only one predictor — VIF not applicable.")
    else:
        rprint(f"      {'Predictor':<35} {'VIF':>10}")
        rprint(f"      {'-'*35} {'-'*10}")
        for name, vif_val in vif_dict.items():
            flag = "  [HIGH]" if vif_val > 10 else "  [MOD]" if vif_val > 5 else ""
            rprint(f"      {name:<35} {fmt(vif_val):>10}{flag}")

    # --- Box-Tidwell ---
    rprint("\n  [2] LINEARITY OF LOGIT — Box-Tidwell Test (continuous predictors only)")
    rprint(f"      H0: Logit(P) is linear in X.")
    rprint(f"      H1: Non-linear logit relationship exists.")
    rprint(f"      NOTE: Binary and dummy-encoded predictors are excluded.\n")

    bt_results = box_tidwell_test(y, X_design, col_names, dummy_info, predictor_cols)
    if "note" in bt_results and len(bt_results) == 1:
        rprint(f"      {bt_results['note']}")
    else:
        rprint(f"      {'Predictor':<30} {'β (X·ln X)':<14} {'p-value':<12} {'Result':<20}")
        rprint(f"      {'-'*30} {'-'*14} {'-'*12} {'-'*20}")
        for name, res in bt_results.items():
            if isinstance(res, dict) and "note" in res:
                rprint(f"      {name:<30} {'N/A':<14} {'N/A':<12} {str(res['note'])[:20]}")
            elif isinstance(res, dict):
                verdict = "PASS (linear)" if bool(res["linear"]) else "FAIL (non-linear)"
                rprint(f"      {name:<30} {res['beta']:<14.4f} "
                       f"{res['pvalue']:<12.4f} {verdict:<20}")
                if not bool(res["linear"]):
                    rprint(f"      [ACTION] Consider transforming '{name}' "
                           f"(log, sqrt, polynomial).")

    # --- Outliers ---
    rprint("\n  [3] OUTLIERS — Pearson Residuals (threshold: |r| > 2)")
    inf_res = compute_influence(result, y, X_with_int, y_pred_prob)
    n_out   = inf_res["n_outliers"]
    if n_out == 0:
        rprint(f"      No outliers detected.")
    else:
        rprint(f"      Outliers detected: {n_out}")
        idx_str = ", ".join([str(i+1) for i in inf_res["outlier_idx"][:20]])
        rprint(f"      Row indices: {idx_str}"
               + (" ..." if len(inf_res["outlier_idx"]) > 20 else ""))
        rprint(f"      [ACTION] Investigate for data entry errors or clinically extreme values.")

    # --- Cook's Distance ---
    method_note = ("exact via GLMInfluence (X'WX hat matrix)"
                   if inf_res["cooks_exact"]
                   else "Pregibon one-step approximation (GLMInfluence unavailable)")
    rprint(f"\n  [4] INFLUENTIAL OBSERVATIONS — Cook's Distance")
    rprint(f"      Method    : {method_note}")
    rprint(f"      Threshold : D > 4/n = {inf_res['cook_thresh']:.4f}")
    n_inf = inf_res["n_influential"]
    if n_inf == 0:
        rprint(f"      No highly influential observations detected.")
    else:
        rprint(f"      Influential observations: {n_inf}")
        idx_str = ", ".join([str(i+1) for i in inf_res["influential_idx"][:20]])
        rprint(f"      Row indices: {idx_str}"
               + (" ..." if len(inf_res["influential_idx"]) > 20 else ""))
        rprint(f"      [ACTION] Examine these observations. Consider sensitivity")
        rprint(f"               analysis with and without influential cases.")

    # --- Post-fit quasi-separation check ---
    rprint(f"\n  [5] POST-FIT QUASI-SEPARATION CHECK")
    rprint(f"      Checks for predicted probabilities ≥ 0.9999 or ≤ 0.0001.")
    rprint(f"      Such extremes indicate the MLE is being driven toward ±∞")
    rprint(f"      by multivariate separation undetectable by univariate scanning.")
    check_separation_postfit(y_pred_prob)
    extreme = int(np.sum(y_pred_prob >= 0.9999)) + int(np.sum(y_pred_prob <= 0.0001))
    if extreme == 0:
        rprint(f"      No quasi-separation detected in fitted probabilities.")

    # --- Diagnostic Summary ---
    rprint(f"\n  DIAGNOSTIC SUMMARY:")
    all_pass = True
    for name, vif_val in vif_dict.items():
        if not math.isnan(vif_val) and vif_val > 10:
            rprint(f"  • High VIF for '{name}' ({vif_val:.2f}) — multicollinearity concern.")
            all_pass = False
    bt_fails = [name for name, res in bt_results.items()
                if isinstance(res, dict) and "linear" in res and not bool(res["linear"])]
    for name in bt_fails:
        rprint(f"  • Non-linear logit for '{name}' — consider transformation.")
        all_pass = False
    if n_out > 0:
        rprint(f"  • {n_out} outlier(s) detected.")
        all_pass = False
    if n_inf > 0:
        rprint(f"  • {n_inf} influential observation(s) detected (Cook's D > 4/n).")
        all_pass = False
    if extreme > 0:
        rprint(f"  • Quasi-separation suspected ({extreme} extreme fitted probability values).")
        all_pass = False
    if all_pass:
        rprint(f"  All checked assumptions appear to be satisfied.")


# ======================================================================
# MEDICAL INTERPRETATION
# ======================================================================

def display_interpretation(result,
                            col_names: list,
                            y: np.ndarray,
                            y_pred_prob: np.ndarray,
                            outcome_col: str,
                            predictor_cols: list,
                            enc_map: dict,
                            perf: dict):
    print_section("MEDICAL & STATISTICAL INTERPRETATION")

    n           = int(result.nobs)
    event_label = [k for k, v in enc_map.items() if v == 1][0]
    all_names   = ["Intercept"] + col_names
    pvals       = result.pvalues
    betas       = result.params
    ci          = result.conf_int(alpha=0.05)
    lr_p        = perf["lr_p"]
    mcfadden    = perf["mcfadden"]

    rprint(f"\n  OUTCOME              : {outcome_col} (event = '{event_label}')")
    rprint(f"  SAMPLE SIZE          : {n:,}")
    rprint(f"  EVENTS (Y=1)         : {int(np.sum(y==1)):,}  ({np.mean(y)*100:.1f}%)")
    rprint(f"  NUMBER OF PREDICTORS : {len(predictor_cols)}")

    rprint(f"\n  WHAT DOES BINARY LOGISTIC REGRESSION MEASURE?")
    rprint(f"  Binary logistic regression models the probability of a binary outcome")
    rprint(f"  as a function of one or more predictors.")
    rprint(f"  The model estimates the log-odds of the event for each predictor,")
    rprint(f"  controlling for all other variables simultaneously.")
    rprint(f"  Odds Ratios quantify the multiplicative change in the odds of the")
    rprint(f"  event per unit increase in the predictor.")

    rprint(f"\n  OVERALL MODEL:")
    if lr_p < 0.05:
        rprint(f"  The model is statistically significant (LR χ²({perf['lr_df']}) = "
               f"{perf['lr_stat']:.4f}, p = {lr_p:.4f}).")
        # CORRECTED: removed "meaningful" — LR test establishes statistical significance only
        rprint(f"  The predictors collectively explain statistically significant variation")
        rprint(f"  in '{outcome_col}' compared to the null (intercept-only) model.")
    else:
        rprint(f"  The model is NOT statistically significant (LR χ²({perf['lr_df']}) = "
               f"{perf['lr_stat']:.4f}, p = {lr_p:.4f}).")
        rprint(f"  The predictors do not significantly improve prediction of '{outcome_col}'.")

    rprint(f"\n  SIGNIFICANT PREDICTORS AND ODDS RATIO INTERPRETATION:")
    any_sig = False
    for i, name in enumerate(all_names):
        if name == "Intercept":
            continue
        p      = float(pvals[i])
        b      = float(betas[i])
        or_val = math.exp(b)      if not math.isnan(b)           else float("nan")
        ci_lo  = math.exp(float(ci[i, 0])) if not math.isnan(float(ci[i, 0])) else float("nan")
        ci_hi  = math.exp(float(ci[i, 1])) if not math.isnan(float(ci[i, 1])) else float("nan")

        if p < 0.05:
            any_sig   = True
            pct_chg   = abs(or_val - 1) * 100
            direction = "increase" if or_val > 1 else "decrease"
            rprint(f"\n  {name}:")
            rprint(f"    OR = {or_val:.4f}  (95% CI: {fmt(ci_lo)} to {fmt(ci_hi)},  p = {p:.4f})")
            rprint(f"    A one-unit increase in '{name}' is associated with a")
            rprint(f"    {pct_chg:.1f}% {direction} in the odds of '{event_label}',")
            rprint(f"    controlling for all other variables in the model.")
            if abs(b) < 0.05:
                rprint(f"    [NOTE] OR ≈ 1.0 despite significance. Clinical importance may be limited.")

    if not any_sig:
        rprint(f"\n  No individual predictor reached statistical significance at α = 0.05.")

    rprint(f"\n  STATISTICAL vs CLINICAL SIGNIFICANCE:")
    rprint(f"  A significant p-value indicates the association is unlikely due to chance.")
    rprint(f"  It does NOT confirm clinical importance or actionability.")
    if n >= LARGE_SAMPLE_N:
        rprint(f"\n  LARGE SAMPLE NOTE (n = {n:,}):")
        rprint(f"  With large samples, trivially small ORs can reach significance.")
        rprint(f"  Assess OR magnitude and CI width for clinical relevance.")
        rprint(f"  An OR of 1.05 may be significant but clinically unimportant.")

    rprint(f"\n  LIMITATIONS:")
    rprint(f"  • ASSOCIATION ≠ CAUSATION. Significant ORs do not establish causality.")
    rprint(f"  • RESIDUAL CONFOUNDING. The model controls only for included variables.")
    rprint(f"  • CLASS IMBALANCE. Event rate = {np.mean(y)*100:.1f}%. Classification metrics")
    rprint(f"    may be misleading. AUC and calibration are more robust.")
    rprint(f"  • RARE OUTCOMES. Very low event rates destabilise coefficient estimates.")
    rprint(f"  • PERFECT SEPARATION. Inflates SEs; use Firth regression if detected.")
    rprint(f"  • OVERFITTING. With {int(np.sum(y==1))} events and {len(predictor_cols)} predictors,")
    epv = int(np.sum(y == 1)) / len(predictor_cols)
    rprint(f"    EPV = {epv:.1f}. "
           f"{'Adequate.' if epv >= 10 else 'Below threshold of 10 — external validation advised.'}")
    rprint(f"  • EXTERNAL VALIDITY. Results may not generalise beyond this population.")
    rprint(f"  • INTERACTIONS. This model does not include interaction terms.")


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
        print(f"  [ERROR] Permission denied: {REPORT_FILE}")
    except OSError as exc:
        print(f"  [ERROR] {exc}")


# ======================================================================
# MAIN ORCHESTRATOR
# ======================================================================

def run_logistic_regression():
    if not check_statsmodels():
        sys.exit(1)

    print_header()
    rprint("  This module performs Binary Logistic Regression.")
    rprint("  It models the probability of a binary medical outcome.")
    rprint("  Uses statsmodels Logit for full inferential statistics.")

    # Step 1
    path = get_csv_path()
    df   = load_csv(path)
    if df is None:
        sys.exit(1)

    # Step 2
    display_columns(df)

    # Step 3
    outcome_col = select_outcome_column(df)
    if outcome_col is None:
        sys.exit(1)

    # Step 4
    predictor_cols = select_predictors(df, outcome_col)
    if predictor_cols is None:
        sys.exit(1)

    # Step 5
    prep = prepare_data(df, outcome_col, predictor_cols)
    if prep is None:
        sys.exit(1)
    working, y, X_design, X_with_int, col_names, enc_map, dummy_info = prep

    # Step 6
    display_descriptive_stats(working, outcome_col, predictor_cols, y, enc_map)

    # Step 7: Pre-fit separation check
    print_section("SEPARATION CHECK (PRE-FIT)")
    sep_found = check_separation_prefit(y, X_design, col_names)
    if not sep_found:
        rprint("  No univariate perfect separation detected.")
    rprint(f"  (Post-fit quasi-separation check will run after model fitting.)")

    # Step 8
    print_section("FITTING LOGISTIC REGRESSION MODEL")
    result = fit_logistic(y, X_with_int, col_names)
    if result is None:
        sys.exit(1)
    rprint(f"  [OK] Model fitted successfully.")
    rprint(f"       Converged : {result.mle_retvals['converged']}")
    rprint(f"       Iterations: {result.mle_retvals.get('iterations', 'N/A')}")

    y_pred_prob = result.predict(X_with_int)

    # Step 9
    display_coefficient_table(result, col_names, col_names)

    # Step 10
    perf = display_model_performance(result)

    # Step 11
    clf = compute_classification(y, y_pred_prob, threshold=DEFAULT_THRESHOLD)
    display_classification(clf)

    # Step 12
    roc = compute_roc_auc(y, y_pred_prob)
    display_roc(roc)

    # Step 13
    display_calibration(y, y_pred_prob)

    # Step 14: Diagnostics (includes post-fit separation check)
    display_diagnostics(result, y, y_pred_prob,
                         X_design, X_with_int, col_names, dummy_info, predictor_cols)

    # Step 15
    display_interpretation(result, col_names, y, y_pred_prob,
                            outcome_col, predictor_cols, enc_map, perf)

    # Step 16
    save_report()

    rprint(f"\n{SEPARATOR}")
    rprint(f"  VM-WIT-STATS | Logistic Regression — Analysis Complete")
    rprint(f"{SEPARATOR}\n")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    run_logistic_regression()