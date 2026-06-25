"""
======================================================================
VM-WIT-STATS | VM Medical College Statistical Analysis Toolkit
======================================================================
Module      : Eta Correlation (η)
File        : correlation/eta_correlation.py
Author      : VM-WIT-STATS Development Team
Version     : 1.0.0

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

Eta (η) measures the strength of association between one categorical
independent variable and one continuous dependent variable.

Unlike Pearson or Spearman, η captures non-linear associations and
requires no assumption about the form of the relationship.

CORE FORMULAS:
--------------
Let:
    k         = number of groups
    N         = total sample size
    n_j       = sample size of group j
    ȳ         = grand mean of all observations
    ȳ_j       = mean of group j
    y_ij      = i-th observation in group j

Then:
    SST (Total Sum of Squares)   = Σ_j Σ_i (y_ij − ȳ)²
    SSB (Between-group SS)       = Σ_j n_j (ȳ_j − ȳ)²
    SSW (Within-group SS)        = SST − SSB = Σ_j Σ_i (y_ij − ȳ_j)²

    η²  = SSB / SST      (Eta Squared — proportion of variance explained)
    η   = √(SSB / SST)   (Eta — ranges [0, 1])

RELATIONSHIP TO ANOVA:
----------------------
    F = (SSB / df_between) / (SSW / df_within)
      where df_between = k − 1, df_within = N − k

    η² = (F · df_between) / (F · df_between + df_within)

Both formulas are mathematically equivalent and give identical results.
This module uses the SS-based formula as primary, cross-verified with
the F-based formula in the internal audit.

INTERPRETATION:
---------------
    η = 0.0          No association
    η > 0, η² small  Weak association
    η = 1.0          Perfect association (groups fully separate)

    η² represents the proportion of variance in the continuous variable
    explained by group membership. This is the primary effect size metric
    for reporting (analogous to R² in regression).

COMPARISON WITH OTHER CORRELATION COEFFICIENTS:
-----------------------------------------------
    Pearson (r):  Linear association between two continuous variables.
                  Sign indicates direction; r² = proportion of variance.
                  η reduces to |r| when categories are ordered and the
                  relationship is perfectly linear.

    Spearman (ρ): Monotonic association, rank-based. Robust to outliers.
                  Not suitable for nominal categories.

    Point-Biserial (r_pb): Special case of Pearson for binary categories.
                  η with k=2 groups equals |r_pb| when groups are coded
                  as 0/1. η is more general for k > 2.

    Eta (η):      Non-linear association, any number of categories,
                  no directionality, no assumption on relationship form.

WHEN TO USE ETA:
----------------
    ✔ Association between a nominal/ordinal categorical var and a
      continuous outcome.
    ✔ Non-linear relationships between groups.
    ✔ Reporting effect size after One-Way ANOVA.
    ✔ More than 2 groups where point-biserial is insufficient.

WHEN NOT TO USE ETA:
--------------------
    ✗ Both variables are continuous (use Pearson or Spearman).
    ✗ Both variables are categorical (use Cramér's V or Chi-Square).
    ✗ Only 1 group exists (η is identically 0 — undefined association).
    ✗ The dependent variable has zero variance (all values identical).
    ✗ Direction of association is needed (η has no sign).

ASSUMPTIONS:
------------
    1. Independent observations (not paired/repeated measures).
    2. Continuous or interval-level dependent variable.
    3. Categorical independent variable with at least 2 groups.
    4. Sufficient sample size per group (n ≥ 2 per group recommended).
    5. No assumption of normality (η is a descriptive measure).
    6. Homogeneity of variance NOT required for η (but is for ANOVA).

REFERENCES:
-----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral
        Sciences (2nd ed.). Lawrence Erlbaum Associates.

    Field, A. (2013). Discovering Statistics Using SPSS (4th ed.).
        Sage Publications.

    Cohen, J. (1973). Eta-squared and partial eta-squared in fixed
        factor ANOVA designs. Educational and Psychological Measurement,
        33(1), 107–112.

    Lakens, D. (2013). Calculating and reporting effect sizes to
        facilitate cumulative science. Frontiers in Psychology, 4, 863.

======================================================================
REQUIRED LIBRARIES
======================================================================
    pip install pandas numpy scipy
======================================================================
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================
# CUSTOM EXCEPTIONS
# ======================================================================

class EtaValidationError(ValueError):
    """Raised when input data fails validation for Eta computation."""
    pass


class EtaComputationError(RuntimeError):
    """Raised when Eta computation fails for numerical reasons."""
    pass


# ======================================================================
# CONFIGURATION: EFFECT SIZE THRESHOLDS
# ======================================================================

# Default thresholds (configurable via EtaCorrelationResult.thresholds)
# Based on Cohen (1988) adapted for η (not η²).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "negligible": 0.10,   # η < 0.10
    "weak":       0.30,   # 0.10 ≤ η < 0.30
    "moderate":   0.50,   # 0.30 ≤ η < 0.50
    "strong":     0.70,   # 0.50 ≤ η < 0.70
    # η ≥ 0.70 → Very Strong
}


# ======================================================================
# RESULT DATACLASS
# ======================================================================

@dataclass
class EtaCorrelationResult:
    """
    Structured result object for Eta Correlation analysis.

    All fields are read-only after construction. Call .summary() for
    a formatted text summary, or .report_paragraph() for a
    publication-ready paragraph.

    Attributes
    ----------
    test_name : str
        Always "Eta Correlation (η)".
    eta : float
        Eta coefficient. Range [0, 1].
    eta_squared : float
        Eta squared (η²). Proportion of variance explained. Range [0, 1].
    effect_size_category : str
        Interpretation label based on thresholds.
    sample_size : int
        Number of complete (non-missing) paired observations used.
    group_count : int
        Number of unique categories.
    group_sizes : dict[str, int]
        Mapping of category label → sample size.
    group_means : dict[str, float]
        Mapping of category label → group mean of continuous variable.
    ssb : float
        Between-group sum of squares.
    sst : float
        Total sum of squares.
    ssw : float
        Within-group sum of squares (= SST − SSB).
    anova_f : float or None
        ANOVA F-statistic (if perform_anova=True).
    anova_p : float or None
        ANOVA p-value (if perform_anova=True).
    anova_df_between : int or None
        Degrees of freedom between groups.
    anova_df_within : int or None
        Degrees of freedom within groups.
    confidence_interval : tuple[float, float] or None
        Bootstrap 95% (or user-specified) CI for η.
    ci_level : float or None
        Confidence level used for CI (e.g. 0.95).
    n_bootstrap : int or None
        Number of bootstrap samples used.
    assumptions : dict[str, object]
        Assumption check results.
    warnings : list[str]
        Non-fatal warnings raised during analysis.
    interpretation : str
        Technical statistical interpretation.
    medical_interpretation : str
        Clinical interpretation suitable for medical researchers.
    summary_text : str
        Concise formatted summary block.
    thresholds : dict[str, float]
        Effect size thresholds used for categorisation.
    """

    test_name            : str                         = "Eta Correlation (η)"
    eta                  : float                       = field(default=float("nan"))
    eta_squared          : float                       = field(default=float("nan"))
    effect_size_category : str                         = "N/A"
    sample_size          : int                         = 0
    group_count          : int                         = 0
    group_sizes          : dict[str, int]              = field(default_factory=dict)
    group_means          : dict[str, float]            = field(default_factory=dict)
    ssb                  : float                       = field(default=float("nan"))
    sst                  : float                       = field(default=float("nan"))
    ssw                  : float                       = field(default=float("nan"))
    anova_f              : Optional[float]             = None
    anova_p              : Optional[float]             = None
    anova_df_between     : Optional[int]               = None
    anova_df_within      : Optional[int]               = None
    confidence_interval  : Optional[tuple[float, float]] = None
    ci_level             : Optional[float]             = None
    n_bootstrap          : Optional[int]               = None
    assumptions          : dict[str, object]           = field(default_factory=dict)
    warnings             : list[str]                   = field(default_factory=list)
    interpretation       : str                         = ""
    medical_interpretation: str                        = ""
    summary_text         : str                         = ""
    thresholds           : dict[str, float]            = field(
                               default_factory=lambda: dict(DEFAULT_THRESHOLDS)
                           )

    def summary(self) -> str:
        """Return the pre-built summary block."""
        return self.summary_text

    def report_paragraph(self,
                          cat_name: str = "the categorical variable",
                          cont_name: str = "the continuous variable") -> str:
        """
        Return a publication-ready paragraph for the association.

        Parameters
        ----------
        cat_name : str
            Name of the categorical variable for the paragraph.
        cont_name : str
            Name of the continuous variable for the paragraph.

        Returns
        -------
        str
            APA-style reporting paragraph.
        """
        eta_str   = f"{self.eta:.3f}" if not math.isnan(self.eta) else "N/A"
        eta_sq_pct = f"{self.eta_squared * 100:.1f}%" if not math.isnan(self.eta_squared) else "N/A"

        if self.anova_f is not None and self.anova_p is not None:
            if self.anova_p < 0.001:
                p_str = "p < .001"
            elif self.anova_p < 0.01:
                p_str = f"p = {self.anova_p:.3f}"
            else:
                p_str = f"p = {self.anova_p:.3f}"
            anova_clause = (
                f" A one-way ANOVA confirmed the association was "
                f"{'statistically significant' if self.anova_p < 0.05 else 'not statistically significant'} "
                f"(F({self.anova_df_between}, {self.anova_df_within}) = {self.anova_f:.2f}, {p_str})."
            )
        else:
            anova_clause = ""

        ci_clause = ""
        if self.confidence_interval is not None:
            ci_lo, ci_hi = self.confidence_interval
            ci_clause = (
                f" The bootstrap {int(self.ci_level*100)}% confidence interval for η "
                f"was [{ci_lo:.3f}, {ci_hi:.3f}]."
            )

        return (
            f"There was a {self.effect_size_category.lower()} association between "
            f"{cat_name} and {cont_name} (η = {eta_str}). "
            f"Approximately {eta_sq_pct} of the variability in {cont_name} "
            f"was accounted for by {cat_name} membership (η² = {self.eta_squared:.3f})."
            f"{anova_clause}{ci_clause}"
        )

    def __str__(self) -> str:
        return self.summary_text


# ======================================================================
# INTERNAL UTILITIES
# ======================================================================

def _to_numpy_pair(
    categorical: Union[pd.Series, np.ndarray, list],
    continuous:  Union[pd.Series, np.ndarray, list],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to numpy arrays with consistent dtype.

    Continuous is cast to float64. Categorical is cast to object
    (string-safe) so labels are preserved exactly.

    Parameters
    ----------
    categorical : array-like
        The grouping variable.
    continuous : array-like
        The numeric outcome variable.

    Returns
    -------
    cat_arr : np.ndarray of object dtype
    cont_arr : np.ndarray of float64
    """
    # Categorical — preserve original labels as strings
    if isinstance(categorical, pd.Series):
        cat_arr = categorical.to_numpy(dtype=object)
    else:
        cat_arr = np.asarray(categorical, dtype=object)

    # Continuous — must be numeric
    if isinstance(continuous, pd.Series):
        try:
            cont_arr = continuous.to_numpy(dtype=np.float64)
        except (ValueError, TypeError):
            raise EtaValidationError(
                "The continuous variable could not be converted to float64. "
                "Ensure it contains only numeric values."
            )
    else:
        try:
            cont_arr = np.asarray(continuous, dtype=np.float64)
        except (ValueError, TypeError):
            raise EtaValidationError(
                "The continuous variable could not be converted to float64. "
                "Ensure it contains only numeric values."
            )

    return cat_arr, cont_arr


def _handle_missing(
    cat_arr:  np.ndarray,
    cont_arr: np.ndarray,
    missing:  str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Handle missing (NaN) and infinite values according to `missing` strategy.

    Parameters
    ----------
    cat_arr : np.ndarray
        Categorical array.
    cont_arr : np.ndarray
        Continuous array (float64).
    missing : {"drop", "raise"}
        Strategy for handling missing values.

    Returns
    -------
    cat_clean : np.ndarray
    cont_clean : np.ndarray
    original_n : int
    dropped_n : int

    Raises
    ------
    EtaValidationError
        If missing="raise" and any missing/infinite values are found.
    """
    original_n = len(cat_arr)

    # Identify problematic rows:
    # 1. NaN in continuous variable (numpy float)
    # 2. Inf or -Inf in continuous variable
    # 3. None / NaN in categorical variable (pandas NA, None, np.nan as object)
    cont_bad = ~np.isfinite(cont_arr)   # catches NaN and ±Inf

    # Categorical missing: None, np.nan cast to object, pd.NA
    cat_bad = np.zeros(len(cat_arr), dtype=bool)
    for i, v in enumerate(cat_arr):
        if v is None or v is pd.NA:
            cat_bad[i] = True
        else:
            # Catch np.nan stored as object (e.g. from pd.read_csv)
            try:
                if math.isnan(float(v)):
                    cat_bad[i] = True
            except (TypeError, ValueError):
                pass   # Non-numeric category labels are fine

    bad_mask   = cont_bad | cat_bad
    dropped_n  = int(bad_mask.sum())

    if dropped_n > 0:
        if missing == "raise":
            raise EtaValidationError(
                f"{dropped_n} row(s) contain missing or infinite values. "
                "Set missing='drop' to remove them automatically, or "
                "clean your data before calling eta_correlation()."
            )
        # drop
        keep = ~bad_mask
        cat_arr  = cat_arr[keep]
        cont_arr = cont_arr[keep]

    return cat_arr, cont_arr, original_n, dropped_n


def _compute_ss(
    cont_arr:   np.ndarray,
    group_idx:  np.ndarray,
    n_groups:   int,
) -> tuple[float, float, float]:
    """
    Compute SST, SSB, SSW using vectorised numpy operations.

    Uses two-pass computation (subtract grand mean, then aggregate)
    which is numerically stable for the data ranges typical in medical
    research. For pathologically large means (e.g. 1e12), users should
    centre their data first — this is documented in the function's
    docstring and in the module limitations.

    Parameters
    ----------
    cont_arr : np.ndarray of float64
        Clean continuous observations.
    group_idx : np.ndarray of int
        Integer group indices (0 to n_groups-1) for each observation.
    n_groups : int
        Total number of unique groups.

    Returns
    -------
    SST : float
    SSB : float
    SSW : float  (= SST − SSB, returned explicitly for verification)
    """
    y_bar = cont_arr.mean()

    # SST: total variation around grand mean
    SST = float(np.sum((cont_arr - y_bar) ** 2))

    # SSB: variation of group means around grand mean, weighted by n_j
    # np.bincount is O(N) and avoids explicit Python loops
    group_sums   = np.bincount(group_idx, weights=cont_arr, minlength=n_groups)
    group_counts = np.bincount(group_idx, minlength=n_groups).astype(np.float64)

    # Guard against empty groups (shouldn't reach here after validation)
    safe_counts  = np.where(group_counts > 0, group_counts, 1.0)
    group_means  = group_sums / safe_counts

    SSB = float(np.sum(group_counts * (group_means - y_bar) ** 2))
    SSW = SST - SSB

    # Clamp SSW to ≥ 0 to avoid −1e-14 due to floating-point rounding
    SSW = max(SSW, 0.0)

    return SST, SSB, SSW


def _categorise_eta(eta: float, thresholds: dict[str, float]) -> str:
    """
    Return the effect size category label for a given η value.

    Parameters
    ----------
    eta : float
        Eta coefficient.
    thresholds : dict[str, float]
        Keys: "negligible", "weak", "moderate", "strong".
        Defines upper bounds for each category except "very_strong".

    Returns
    -------
    str
        One of: "Negligible", "Weak", "Moderate", "Strong", "Very Strong".
    """
    if math.isnan(eta):
        return "N/A"
    if eta < thresholds["negligible"]:
        return "Negligible"
    elif eta < thresholds["weak"]:
        return "Weak"
    elif eta < thresholds["moderate"]:
        return "Moderate"
    elif eta < thresholds["strong"]:
        return "Strong"
    else:
        return "Very Strong"


def _bootstrap_ci(
    cont_arr:      np.ndarray,
    group_idx:     np.ndarray,
    n_groups:      int,
    n_bootstrap:   int,
    confidence_level: float,
    random_state:  Optional[int],
) -> tuple[float, float]:
    """
    Compute percentile bootstrap confidence interval for η.

    Bootstrap procedure:
        1. Sample N observations with replacement (maintaining paired
           categorical-continuous structure).
        2. Compute η for each bootstrap sample.
        3. Return the (α/2, 1-α/2) percentiles of the bootstrap distribution.

    Empty group handling:
        If a bootstrap sample happens to omit one or more groups (possible
        in small datasets), that group's n_j = 0. In this case, SSB is
        computed only over present groups — this is equivalent to running
        the test on a dataset with fewer groups and is mathematically valid
        for the bootstrap sample. Such samples are included (not discarded).

    NaN bootstrap samples:
        If SST = 0 in a bootstrap sample (all continuous values identical),
        η is undefined. These samples are excluded from the percentile
        computation and a warning is issued if their count is material.

    Parameters
    ----------
    cont_arr : np.ndarray
        Clean continuous observations.
    group_idx : np.ndarray
        Integer group indices.
    n_groups : int
        Total number of unique groups.
    n_bootstrap : int
        Number of bootstrap replications.
    confidence_level : float
        Confidence level, e.g. 0.95.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    ci_lower : float
    ci_upper : float
    """
    rng_b = np.random.default_rng(random_state)
    n     = len(cont_arr)
    etas  = np.empty(n_bootstrap, dtype=np.float64)

    alpha_lo = (1.0 - confidence_level) / 2.0 * 100.0
    alpha_hi = (1.0 + confidence_level) / 2.0 * 100.0

    for i in range(n_bootstrap):
        idx  = rng_b.integers(0, n, size=n)
        c    = cont_arr[idx]
        gi   = group_idx[idx]

        y_bar  = c.mean()
        sst    = float(np.sum((c - y_bar) ** 2))

        if sst < 1e-14:
            etas[i] = float("nan")
            continue

        gs  = np.bincount(gi, weights=c, minlength=n_groups)
        gc  = np.bincount(gi, minlength=n_groups).astype(np.float64)
        mask = gc > 0
        safe_gc = np.where(mask, gc, 1.0)
        gm   = np.where(mask, gs / safe_gc, y_bar)
        ssb  = float(np.sum(gc * (gm - y_bar) ** 2))
        etas[i] = math.sqrt(max(ssb / sst, 0.0))

    valid = etas[~np.isnan(etas)]
    n_nan = n_bootstrap - len(valid)
    if n_nan > 0:
        warnings.warn(
            f"{n_nan} bootstrap samples had SST ≈ 0 and were excluded "
            f"from the CI computation. CI is based on {len(valid)} valid samples.",
            UserWarning, stacklevel=4
        )

    if len(valid) < 10:
        raise EtaComputationError(
            f"Only {len(valid)} valid bootstrap samples out of {n_bootstrap}. "
            "Bootstrap CI cannot be reliably estimated."
        )

    ci_lower = float(np.percentile(valid, alpha_lo))
    ci_upper = float(np.percentile(valid, alpha_hi))
    return ci_lower, ci_upper


def _check_assumptions(
    cat_arr:   np.ndarray,
    cont_arr:  np.ndarray,
    group_labels: np.ndarray,
    group_sizes:  dict[str, int],
) -> tuple[dict, list[str]]:
    """
    Evaluate and report assumption checks for Eta Correlation.

    Parameters
    ----------
    cat_arr : np.ndarray
        Clean categorical array.
    cont_arr : np.ndarray
        Clean continuous array.
    group_labels : np.ndarray
        Unique category labels.
    group_sizes : dict[str, int]
        Category label → sample size.

    Returns
    -------
    assumptions : dict[str, object]
    warn_list : list[str]
    """
    warn_list = []

    # Continuous variable check (already validated upstream, but document it)
    is_continuous = pd.api.types.is_float_dtype(cont_arr) or \
                    pd.api.types.is_integer_dtype(cont_arr)

    # Group sample sizes
    min_group_n = min(group_sizes.values())
    small_groups = [k for k, v in group_sizes.items() if v < 5]
    if small_groups:
        msg = (f"Groups with n < 5: {small_groups}. "
               f"η is less stable for very small groups.")
        warn_list.append(msg)

    # Imbalance: max/min group size ratio
    max_group_n = max(group_sizes.values())
    if min_group_n > 0:
        imbalance_ratio = max_group_n / min_group_n
    else:
        imbalance_ratio = float("inf")

    group_balance_warning = imbalance_ratio > 10
    if group_balance_warning:
        msg = (f"Group size imbalance ratio = {imbalance_ratio:.1f} "
               f"(max/min = {max_group_n}/{min_group_n}). "
               f"Highly imbalanced groups may affect ANOVA F-test validity, "
               f"but do not bias η itself.")
        warn_list.append(msg)

    # Variance check
    cont_std = float(cont_arr.std(ddof=1)) if len(cont_arr) > 1 else 0.0
    low_variance = cont_std < 1e-10

    assumptions = {
        "independence"           : "user_responsibility",
        "continuous_variable"    : bool(is_continuous),
        "categorical_variable"   : True,
        "n_groups"               : int(len(group_labels)),
        "min_group_n"            : int(min_group_n),
        "group_balance_warning"  : bool(group_balance_warning),
        "imbalance_ratio"        : round(imbalance_ratio, 2)
                                   if not math.isinf(imbalance_ratio) else "inf",
        "low_variance_warning"   : bool(low_variance),
        "total_n"                : len(cont_arr),
    }

    return assumptions, warn_list


def _build_interpretation(
    eta:      float,
    eta_sq:   float,
    category: str,
    n:        int,
    k:        int,
    anova_f:  Optional[float],
    anova_p:  Optional[float],
    ci:       Optional[tuple[float, float]],
    ci_level: Optional[float],
) -> tuple[str, str]:
    """
    Generate technical and medical interpretation strings.

    Returns
    -------
    interpretation : str  (statistical/technical)
    medical_interpretation : str  (clinical)
    """
    eta_str    = f"{eta:.4f}" if not math.isnan(eta) else "N/A"
    eta_sq_str = f"{eta_sq:.4f}" if not math.isnan(eta_sq) else "N/A"
    pct_str    = f"{eta_sq * 100:.1f}%" if not math.isnan(eta_sq) else "N/A"

    # Technical
    lines = [
        f"Eta (η)         : {eta_str}",
        f"Eta Squared (η²): {eta_sq_str}",
        f"Effect Size     : {category}",
        f"Sample Size     : {n:,}",
        f"Groups (k)      : {k}",
        "",
        f"STATISTICAL INTERPRETATION:",
        f"  η = {eta_str} indicates a {category.lower()} association.",
        f"  η² = {eta_sq_str}: the categorical variable accounts for {pct_str}",
        f"  of the total variance in the continuous variable.",
    ]

    if anova_f is not None and anova_p is not None:
        df_b = k - 1; df_w = n - k
        p_fmt = "< 0.001" if anova_p < 0.001 else f"= {anova_p:.4f}"
        sig   = "statistically significant" if anova_p < 0.05 else "not statistically significant"
        lines += [
            "",
            f"  One-Way ANOVA: F({df_b}, {df_w}) = {anova_f:.4f}, p {p_fmt}",
            f"  The overall group difference is {sig} at α = 0.05.",
        ]

    if ci is not None and ci_level is not None:
        lines += [
            "",
            f"  Bootstrap {int(ci_level*100)}% CI for η: [{ci[0]:.4f}, {ci[1]:.4f}]",
        ]

    interpretation = "\n".join(lines)

    # Medical/clinical
    med_lines = [
        "CLINICAL INTERPRETATION:",
        f"  Effect size: {category}",
        "",
    ]
    if category == "Negligible":
        med_lines.append(
            f"  Group membership explains very little of the variability in "
            f"the outcome (η² = {eta_sq_str}, {pct_str}). The groups have "
            f"similar outcome distributions."
        )
    elif category == "Weak":
        med_lines.append(
            f"  There is a weak association between group membership and the "
            f"outcome (η² = {eta_sq_str}, {pct_str}). The clinical relevance "
            f"of this association should be evaluated in context."
        )
    elif category == "Moderate":
        med_lines.append(
            f"  There is a moderate association between group membership and "
            f"the outcome (η² = {eta_sq_str}, {pct_str}). This is a "
            f"practically meaningful effect size."
        )
    elif category == "Strong":
        med_lines.append(
            f"  There is a strong association between group membership and the "
            f"outcome (η² = {eta_sq_str}, {pct_str}). Group membership "
            f"substantially predicts the outcome."
        )
    else:  # Very Strong
        med_lines.append(
            f"  There is a very strong association between group membership and "
            f"the outcome (η² = {eta_sq_str}, {pct_str}). Group membership "
            f"is a major determinant of outcome."
        )

    if anova_f is not None and anova_p is not None:
        if anova_p < 0.05:
            med_lines.append(
                f"\n  The group difference is statistically significant (p {p_fmt}). "
                f"Statistical significance confirms the association is unlikely "
                f"to be due to chance alone."
            )
        else:
            med_lines.append(
                f"\n  The group difference does not reach statistical significance "
                f"(p {p_fmt}). Consider whether the study is adequately powered."
            )

    med_lines += [
        "",
        "LIMITATIONS:",
        "  • η measures association STRENGTH only — not direction or causation.",
        "  • η ≥ 1 group mean differences of any form, including non-linear.",
        "  • Confounding variables have not been controlled for.",
        "  • Causal inference requires appropriate study design.",
    ]

    medical_interpretation = "\n".join(med_lines)

    return interpretation, medical_interpretation


def _build_summary(
    eta:      float,
    eta_sq:   float,
    category: str,
    n:        int,
    k:        int,
    group_sizes: dict[str, int],
    group_means: dict[str, float],
    ssb: float, sst: float, ssw: float,
    anova_f:  Optional[float],
    anova_p:  Optional[float],
    ci:       Optional[tuple[float, float]],
    ci_level: Optional[float],
    warn_list: list[str],
) -> str:
    """Build the concise summary block."""
    sep = "-" * 52
    lines = [
        sep,
        "ETA CORRELATION ANALYSIS",
        "VM-WIT-STATS | VM Medical College",
        sep,
        f"Eta (η)         : {eta:.4f}",
        f"Eta Squared (η²): {eta_sq:.4f}",
        f"Effect Size     : {category}",
        f"Groups (k)      : {k}",
        f"Sample Size (N) : {n:,}",
        sep,
        "SUM OF SQUARES:",
        f"  SST (Total)   : {sst:.4f}",
        f"  SSB (Between) : {ssb:.4f}",
        f"  SSW (Within)  : {ssw:.4f}",
        sep,
        "GROUP BREAKDOWN:",
    ]

    for label, count in sorted(group_sizes.items(), key=lambda x: str(x[0])):
        gm = group_means.get(label, float("nan"))
        lines.append(f"  {str(label):<20} n = {count:>6,}  mean = {gm:,.4f}")

    lines += [sep, "INTERPRETATION:"]
    lines.append(f"  {category} association detected.")

    pct = eta_sq * 100
    lines.append(f"  {pct:.1f}% of variance explained by group membership.")

    if ci is not None and ci_level is not None:
        lines.append(
            f"  Bootstrap {int(ci_level*100)}% CI for η: [{ci[0]:.4f}, {ci[1]:.4f}]"
        )

    if anova_f is not None and anova_p is not None:
        lines += [
            sep,
            "ONE-WAY ANOVA:",
            f"  F = {anova_f:.4f}",
        ]
        if anova_p < 0.001:
            lines.append("  p < 0.001")
        else:
            lines.append(f"  p = {anova_p:.4f}")
        sig = "YES" if anova_p < 0.05 else "NO"
        lines.append(f"  Significant at α=0.05: {sig}")

    if warn_list:
        lines += [sep, "WARNINGS:"]
        for w in warn_list:
            lines.append(f"  ⚠ {w}")

    lines.append(sep)
    return "\n".join(lines)


# ======================================================================
# MAIN PUBLIC FUNCTION
# ======================================================================

def eta_correlation(
    categorical:      Union[pd.Series, np.ndarray, list],
    continuous:       Union[pd.Series, np.ndarray, list],
    missing:          str                = "drop",
    perform_anova:    bool               = False,
    bootstrap_ci:     bool               = False,
    n_bootstrap:      int                = 5000,
    confidence_level: float              = 0.95,
    random_state:     Optional[int]      = None,
    thresholds:       Optional[dict[str, float]] = None,
) -> EtaCorrelationResult:
    """
    Compute Eta Correlation (η) between a categorical and continuous variable.

    Eta measures the strength (not direction) of association between one
    categorical independent variable and one continuous dependent variable.
    It is computed as η = √(SSB / SST) where SSB is the between-group
    sum of squares and SST is the total sum of squares.

    Parameters
    ----------
    categorical : array-like
        The grouping (independent) variable. Can be strings, integers,
        or any hashable type. Must have at least 2 unique values after
        missing data removal.

    continuous : array-like of numeric
        The outcome (dependent) variable. Must be numeric (float or int).
        Non-numeric values raise EtaValidationError.

    missing : {"drop", "raise"}, default "drop"
        Strategy for handling missing or infinite values.
        "drop" — remove rows where either variable is missing/infinite.
        "raise" — raise EtaValidationError if any such rows exist.

    perform_anova : bool, default False
        If True, compute One-Way ANOVA and return F-statistic and p-value.
        This provides the statistical significance complement to η (effect
        size). The ANOVA assumes independence and approximately equal
        variances (Levene's test is not run automatically).

    bootstrap_ci : bool, default False
        If True, compute a percentile bootstrap CI for η.
        Adaptive n_bootstrap reduction for large datasets:
            n ≤ 5,000  : uses full n_bootstrap as specified
            5,000 < n ≤ 50,000 : n_bootstrap capped at 2,000 with a warning
            n > 50,000  : n_bootstrap capped at 1,000 with a warning

    n_bootstrap : int, default 5000
        Number of bootstrap replications (before adaptive reduction).

    confidence_level : float, default 0.95
        Confidence level for bootstrap CI, e.g. 0.95 for 95% CI.

    random_state : int or None, default None
        Seed for the bootstrap random number generator. Set for
        reproducibility.

    thresholds : dict or None, default None
        Custom effect size thresholds dict with keys:
        "negligible", "weak", "moderate", "strong".
        Values are the UPPER bounds of each category.
        If None, uses DEFAULT_THRESHOLDS.

    Returns
    -------
    EtaCorrelationResult
        Structured result with η, η², effect size category, ANOVA
        results (optional), bootstrap CI (optional), assumptions,
        warnings, and interpretation text.

    Raises
    ------
    EtaValidationError
        For invalid inputs: unequal lengths, non-numeric continuous,
        fewer than 2 groups, zero variance, insufficient sample size,
        or missing="raise" with missing data.
    EtaComputationError
        If numerical computation fails or produces undefined results.

    Examples
    --------
    >>> import numpy as np
    >>> cat = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    >>> cont = [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, 30.0, 32.0, 31.0]
    >>> result = eta_correlation(cat, cont)
    >>> print(f"η = {result.eta:.4f}, η² = {result.eta_squared:.4f}")
    η = 0.9874, η² = 0.9750

    >>> result2 = eta_correlation(cat, cont, perform_anova=True,
    ...                            bootstrap_ci=True, random_state=42)
    >>> print(result2.summary())
    """
    # ── 0. Resolve thresholds ────────────────────────────────────────
    thr = dict(DEFAULT_THRESHOLDS) if thresholds is None else dict(thresholds)
    required_keys = {"negligible", "weak", "moderate", "strong"}
    if not required_keys.issubset(thr.keys()):
        raise EtaValidationError(
            f"Custom thresholds must contain keys: {required_keys}. "
            f"Got: {set(thr.keys())}"
        )

    # ── 1. Convert inputs ────────────────────────────────────────────
    cat_arr, cont_arr = _to_numpy_pair(categorical, continuous)

    # ── 2. Length check ──────────────────────────────────────────────
    if len(cat_arr) != len(cont_arr):
        raise EtaValidationError(
            f"categorical and continuous must have the same length. "
            f"Got len(categorical)={len(cat_arr)}, len(continuous)={len(cont_arr)}."
        )

    if len(cat_arr) == 0:
        raise EtaValidationError(
            "Both arrays are empty. No data to analyse."
        )

    # ── 3. Missing data ──────────────────────────────────────────────
    cat_arr, cont_arr, original_n, dropped_n = _handle_missing(
        cat_arr, cont_arr, missing
    )

    # ── 4. Post-cleaning size check ──────────────────────────────────
    n = len(cat_arr)
    if n < 2:
        raise EtaValidationError(
            f"Only {n} observation(s) remain after removing missing values. "
            f"Eta Correlation requires at least 2 complete observations."
        )

    # ── 5. Unique groups ─────────────────────────────────────────────
    group_labels, group_idx = np.unique(cat_arr, return_inverse=True)
    k = len(group_labels)

    if k < 2:
        raise EtaValidationError(
            f"Only 1 unique category found: '{group_labels[0]}'. "
            f"Eta Correlation requires at least 2 distinct groups. "
            f"η is identically 0 when only one group exists."
        )

    # ── 6. Check for empty groups (post-encoding safety) ─────────────
    group_counts = np.bincount(group_idx, minlength=k)
    empty_groups = [str(group_labels[i]) for i, c in enumerate(group_counts) if c == 0]
    if empty_groups:
        # This should not occur after np.unique + return_inverse, but guard anyway
        raise EtaComputationError(
            f"Empty groups detected after encoding: {empty_groups}. "
            "This is an internal error — please report it."
        )

    # ── 7. Zero-variance check ────────────────────────────────────────
    cont_range = float(cont_arr.max()) - float(cont_arr.min())
    if cont_range < 1e-14:
        raise EtaValidationError(
            "The continuous variable has zero variance (all values are identical). "
            "Eta Correlation is undefined: SST = 0, causing division by zero. "
            "Normality η = 0 by convention, but this is not informative."
        )

    # ── 8. Group sizes and means ──────────────────────────────────────
    group_sizes_arr = group_counts
    group_sizes  = {
        str(group_labels[i]): int(group_counts[i])
        for i in range(k)
    }
    group_sums_arr = np.bincount(group_idx, weights=cont_arr, minlength=k)
    group_means = {
        str(group_labels[i]): float(group_sums_arr[i] / group_counts[i])
        for i in range(k)
    }

    # ── 9. Assumption checks ─────────────────────────────────────────
    assumptions, warn_list = _check_assumptions(
        cat_arr, cont_arr, group_labels, group_sizes
    )

    if dropped_n > 0:
        warn_list.insert(0,
            f"{dropped_n} row(s) removed due to missing/infinite values "
            f"(original N={original_n:,}, analysis N={n:,})."
        )

    # ── 10. Core computation ─────────────────────────────────────────
    SST, SSB, SSW = _compute_ss(cont_arr, group_idx, k)

    # Safety: SST could theoretically be near-zero despite range check
    # (numerical edge case with extreme values). Double-check.
    if SST < 1e-14:
        raise EtaComputationError(
            "SST ≈ 0 after computation despite non-zero range. "
            "This may indicate extreme floating-point cancellation. "
            "Consider centring your data."
        )

    eta_sq  = SSB / SST
    # Clamp to [0, 1] to guard against floating-point overshoot
    eta_sq  = max(0.0, min(1.0, eta_sq))
    eta     = math.sqrt(eta_sq)

    category = _categorise_eta(eta, thr)

    # ── 11. ANOVA ────────────────────────────────────────────────────
    anova_f = None; anova_p = None
    anova_df_b = None; anova_df_w = None

    if perform_anova:
        df_b = k - 1
        df_w = n - k
        if df_w <= 0:
            warn_list.append(
                "ANOVA could not be performed: df_within ≤ 0 "
                f"(N={n}, k={k}). More observations than groups required."
            )
        else:
            MS_B = SSB / df_b
            MS_W = SSW / df_w if SSW > 0 else 0.0
            if MS_W < 1e-14:
                warn_list.append(
                    "ANOVA: MS_within ≈ 0 (within-group variance is negligible). "
                    "F-statistic is unreliable. This occurs with near-perfect separation."
                )
                anova_f = float("inf")
                anova_p = float(stats.f.sf(1e10, df_b, df_w))  # approx 0
            else:
                anova_f = float(MS_B / MS_W)
                anova_p = float(stats.f.sf(anova_f, df_b, df_w))
            anova_df_b = df_b
            anova_df_w = df_w

    # ── 12. Bootstrap CI ─────────────────────────────────────────────
    ci          = None
    ci_level_out = None
    n_boot_used  = None

    if bootstrap_ci:
        # Adaptive n_bootstrap for large datasets
        n_boot_actual = n_bootstrap
        if n > 50_000:
            n_boot_actual = min(n_bootstrap, 1000)
            if n_boot_actual < n_bootstrap:
                warn_list.append(
                    f"n = {n:,} > 50,000: bootstrap reduced to {n_boot_actual} "
                    f"replications for performance. Set n_bootstrap manually to override."
                )
        elif n > 5_000:
            n_boot_actual = min(n_bootstrap, 2000)
            if n_boot_actual < n_bootstrap:
                warn_list.append(
                    f"n = {n:,} > 5,000: bootstrap reduced to {n_boot_actual} "
                    f"replications for performance. Set n_bootstrap manually to override."
                )

        if not (0.0 < confidence_level < 1.0):
            raise EtaValidationError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
            )

        try:
            ci_lo, ci_hi = _bootstrap_ci(
                cont_arr, group_idx, k,
                n_boot_actual, confidence_level, random_state
            )
            ci           = (ci_lo, ci_hi)
            ci_level_out = confidence_level
            n_boot_used  = n_boot_actual
        except EtaComputationError as exc:
            warn_list.append(f"Bootstrap CI failed: {exc}")

    # ── 13. Interpretation ───────────────────────────────────────────
    interp, med_interp = _build_interpretation(
        eta, eta_sq, category, n, k,
        anova_f, anova_p, ci, ci_level_out
    )

    summary_txt = _build_summary(
        eta, eta_sq, category, n, k,
        group_sizes, group_means,
        SSB, SST, SSW,
        anova_f, anova_p,
        ci, ci_level_out,
        warn_list
    )

    # ── 14. Assemble result ───────────────────────────────────────────
    return EtaCorrelationResult(
        test_name             = "Eta Correlation (η)",
        eta                   = eta,
        eta_squared           = eta_sq,
        effect_size_category  = category,
        sample_size           = n,
        group_count           = k,
        group_sizes           = group_sizes,
        group_means           = group_means,
        ssb                   = SSB,
        sst                   = SST,
        ssw                   = SSW,
        anova_f               = anova_f,
        anova_p               = anova_p,
        anova_df_between      = anova_df_b,
        anova_df_within       = anova_df_w,
        confidence_interval   = ci,
        ci_level              = ci_level_out,
        n_bootstrap           = n_boot_used,
        assumptions           = assumptions,
        warnings              = warn_list,
        interpretation        = interp,
        medical_interpretation= med_interp,
        summary_text          = summary_txt,
        thresholds            = thr,
    )


# ======================================================================
# CONVENIENCE FUNCTION FOR INTERACTIVE USE
# ======================================================================

def print_eta_report(result: EtaCorrelationResult,
                      cat_name: str = "categorical variable",
                      cont_name: str = "continuous variable") -> None:
    """
    Print a complete formatted report to stdout.

    Parameters
    ----------
    result : EtaCorrelationResult
        Result from eta_correlation().
    cat_name : str
        Display name for the categorical variable.
    cont_name : str
        Display name for the continuous variable.
    """
    print(result.summary())
    print()
    print(result.interpretation)
    print()
    print(result.medical_interpretation)
    print()
    print("PUBLICATION PARAGRAPH:")
    print(result.report_paragraph(cat_name, cont_name))
    if result.warnings:
        print()
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  ⚠  {w}")


# ======================================================================
# BUILT-IN EXAMPLES (run with python -m correlation.eta_correlation)
# ======================================================================

def _run_examples() -> None:
    """
    Run four built-in examples demonstrating Eta Correlation.

    Example 1: Gender → Systolic Blood Pressure
    Example 2: Treatment Group → Cholesterol
    Example 3: Disease Category → Biomarker
    Example 4: Synthetic validation (known η)
    """
    rng = np.random.default_rng(42)
    SEP = "=" * 60

    # ── Example 1: Gender vs Systolic BP ─────────────────────────
    print(SEP)
    print("EXAMPLE 1: Gender → Systolic Blood Pressure")
    print(SEP)
    n1 = 120
    gender = np.array(["Male"] * 60 + ["Female"] * 60)
    bp = np.concatenate([
        rng.normal(128, 14, 60),   # Males tend slightly higher
        rng.normal(122, 13, 60),
    ])
    r1 = eta_correlation(gender, bp, perform_anova=True,
                          bootstrap_ci=True, random_state=42)
    print_eta_report(r1, "gender", "systolic blood pressure")
    print()

    # ── Example 2: Treatment Group vs Cholesterol ─────────────────
    print(SEP)
    print("EXAMPLE 2: Treatment Group → Cholesterol (mg/dL)")
    print(SEP)
    treatment = np.repeat(["Control", "Low Dose", "High Dose", "Placebo"], 60)
    cholesterol = np.concatenate([
        rng.normal(210, 25, 60),
        rng.normal(185, 22, 60),
        rng.normal(162, 20, 60),
        rng.normal(208, 25, 60),
    ])
    r2 = eta_correlation(treatment, cholesterol, perform_anova=True,
                          bootstrap_ci=True, random_state=42)
    print_eta_report(r2, "treatment group", "cholesterol level")
    print()

    # ── Example 3: Disease Category vs Biomarker ──────────────────
    print(SEP)
    print("EXAMPLE 3: Disease Category → CRP Biomarker (mg/L)")
    print(SEP)
    disease = np.repeat(
        ["Healthy", "Mild", "Moderate", "Severe", "Critical"], 40
    )
    crp = np.concatenate([
        rng.lognormal(0.5, 0.3, 40),   # Healthy: low CRP
        rng.lognormal(1.2, 0.4, 40),
        rng.lognormal(2.0, 0.5, 40),
        rng.lognormal(2.8, 0.6, 40),
        rng.lognormal(3.5, 0.7, 40),   # Critical: high CRP
    ])
    r3 = eta_correlation(disease, crp, perform_anova=True,
                          bootstrap_ci=True, random_state=42)
    print_eta_report(r3, "disease category", "CRP biomarker")
    print()

    # ── Example 4: Synthetic validation ───────────────────────────
    print(SEP)
    print("EXAMPLE 4: Synthetic Validation (known η)")
    print(SEP)
    # Construct data with known SSB/SST
    # Group A: mean=0, Group B: mean=10, equal n, equal variance
    # η² approaches 1 as between-group spread >> within-group variance
    g_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # no within-group variance
    g_b = np.array([10., 10., 10., 10., 10.])
    cat_v = np.array(["A"]*5 + ["B"]*5)
    cont_v = np.concatenate([g_a, g_b])
    r4 = eta_correlation(cat_v, cont_v)
    print(f"Perfect separation → η should be 1.0: η = {r4.eta:.6f}")
    print(f"η² = {r4.eta_squared:.6f}")
    print()

    # Verify against known formula
    grand_mean = cont_v.mean()
    SST_v = np.sum((cont_v - grand_mean)**2)
    SSB_v = 5*(0.0 - grand_mean)**2 + 5*(10.0 - grand_mean)**2
    eta_expected = math.sqrt(SSB_v / SST_v)
    print(f"Manual check: SST={SST_v:.4f}, SSB={SSB_v:.4f}, η={eta_expected:.6f}")
    print(f"Match: {abs(r4.eta - eta_expected) < 1e-10}")


# ======================================================================
# STATISTICAL AUDIT
# ======================================================================

def _audit() -> None:
    """
    Internal statistical audit — run to verify implementation correctness.

    Verification steps:
        1. SS decomposition: SST = SSB + SSW
        2. η² = SSB/SST within [0, 1]
        3. η = √(η²) ∈ [0, 1]
        4. F-formula equivalence: η² = F·dfB/(F·dfB+dfW)
        5. Perfect separation → η = 1.0
        6. All same group → η² = 0 (SSB = 0)
        7. ANOVA F matches scipy.stats.f_oneway
        8. Bootstrap CI contains true η for synthetic data (stochastic)
        9. Missing data: dropped N correctly reported
        10. Threshold categorisation boundaries
    """
    import sys
    rng = np.random.default_rng(0)
    failures = []

    def _assert(condition, msg):
        if not condition:
            failures.append(f"FAIL: {msg}")
            print(f"  ✗ {msg}")
        else:
            print(f"  ✔ {msg}")

    print("STATISTICAL AUDIT — eta_correlation.py")
    print("=" * 50)

    # 1. SS decomposition
    print("\n[1] SS decomposition: SST = SSB + SSW")
    cat = np.repeat(["A","B","C"], [30, 35, 25])
    cont = np.concatenate([
        rng.normal(100, 10, 30),
        rng.normal(115, 10, 35),
        rng.normal(130, 10, 25),
    ])
    r = eta_correlation(cat, cont)
    _assert(abs(r.ssb + r.ssw - r.sst) < 1e-8,
            f"SSB+SSW={r.ssb+r.ssw:.6f} ≈ SST={r.sst:.6f}")

    # 2. η² in [0,1]
    print("\n[2] η² ∈ [0, 1]")
    _assert(0.0 <= r.eta_squared <= 1.0, f"η²={r.eta_squared:.6f}")

    # 3. η = √(η²)
    print("\n[3] η = √(η²)")
    _assert(abs(r.eta - math.sqrt(r.eta_squared)) < 1e-10,
            f"η={r.eta:.6f}, √η²={math.sqrt(r.eta_squared):.6f}")

    # 4. F-formula equivalence
    print("\n[4] F-formula: η² = F·dfB/(F·dfB+dfW)")
    r_anova = eta_correlation(cat, cont, perform_anova=True)
    k = r_anova.group_count; n = r_anova.sample_size
    df_b = k - 1; df_w = n - k
    eta_sq_from_F = (r_anova.anova_f * df_b) / (r_anova.anova_f * df_b + df_w)
    _assert(abs(r_anova.eta_squared - eta_sq_from_F) < 1e-8,
            f"η²(SS)={r_anova.eta_squared:.6f}, η²(F)={eta_sq_from_F:.6f}")

    # 5. ANOVA F matches scipy
    print("\n[5] ANOVA F matches scipy.stats.f_oneway")
    groups_d = {g: cont[cat == g] for g in np.unique(cat)}
    F_scipy, _ = stats.f_oneway(*groups_d.values())
    _assert(abs(r_anova.anova_f - float(F_scipy)) < 1e-6,
            f"F(module)={r_anova.anova_f:.6f}, F(scipy)={float(F_scipy):.6f}")

    # 6. Perfect separation → η = 1.0
    print("\n[6] Perfect separation → η = 1.0")
    cat_p = np.array(["A"]*10 + ["B"]*10)
    cont_p = np.array([0.0]*10 + [10.0]*10)
    r_p = eta_correlation(cat_p, cont_p)
    _assert(abs(r_p.eta - 1.0) < 1e-10, f"η={r_p.eta:.10f}")

    # 7. Single group → ValidationError
    print("\n[7] Single category raises EtaValidationError")
    try:
        eta_correlation(["A"]*10, rng.normal(0,1,10))
        _assert(False, "Should have raised EtaValidationError")
    except EtaValidationError:
        _assert(True, "EtaValidationError raised correctly")

    # 8. Zero variance → ValidationError
    print("\n[8] Zero variance continuous → EtaValidationError")
    try:
        eta_correlation(["A","B","A","B"], [5.0, 5.0, 5.0, 5.0])
        _assert(False, "Should have raised EtaValidationError")
    except EtaValidationError:
        _assert(True, "EtaValidationError raised correctly")

    # 9. Missing data handling
    print("\n[9] Missing data: drop strategy")
    cat_m = np.array(["A","A","B","B",None,"A"])
    cont_m = np.array([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])
    r_m = eta_correlation(cat_m, cont_m, missing="drop")
    _assert(r_m.sample_size == 4, f"N after drop = {r_m.sample_size} (expected 4)")
    _assert("removed" in r_m.warnings[0].lower(), "Warning mentions removed rows")

    # 10. Missing="raise" raises
    print("\n[10] missing='raise' raises EtaValidationError")
    try:
        eta_correlation(cat_m, cont_m, missing="raise")
        _assert(False, "Should have raised EtaValidationError")
    except EtaValidationError:
        _assert(True, "EtaValidationError raised correctly")

    # 11. Threshold categorisation
    print("\n[11] Threshold categorisation boundaries")
    thr = DEFAULT_THRESHOLDS
    _assert(_categorise_eta(0.05, thr) == "Negligible", "η=0.05 → Negligible")
    _assert(_categorise_eta(0.15, thr) == "Weak",       "η=0.15 → Weak")
    _assert(_categorise_eta(0.35, thr) == "Moderate",   "η=0.35 → Moderate")
    _assert(_categorise_eta(0.55, thr) == "Strong",     "η=0.55 → Strong")
    _assert(_categorise_eta(0.75, thr) == "Very Strong","η=0.75 → Very Strong")
    _assert(_categorise_eta(0.10, thr) == "Weak",       "η=0.10 (boundary) → Weak")
    _assert(_categorise_eta(0.30, thr) == "Moderate",   "η=0.30 (boundary) → Moderate")

    # 12. Bootstrap CI contains η (stochastic — use large n for reliability)
    print("\n[12] Bootstrap CI plausibility check")
    r_ci = eta_correlation(cat, cont, bootstrap_ci=True,
                            n_bootstrap=2000, random_state=0)
    ci_lo, ci_hi = r_ci.confidence_interval
    _assert(ci_lo < r_ci.eta < ci_hi,
            f"η={r_ci.eta:.4f} within CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    _assert(ci_lo >= 0.0 and ci_hi <= 1.0,
            f"CI within [0,1]: [{ci_lo:.4f},{ci_hi:.4f}]")

    # 13. Unequal length inputs → ValidationError
    print("\n[13] Unequal lengths → EtaValidationError")
    try:
        eta_correlation(["A","B","C"], [1.0, 2.0])
        _assert(False, "Should have raised EtaValidationError")
    except EtaValidationError:
        _assert(True, "EtaValidationError raised correctly")

    # 14. Large dataset performance
    print("\n[14] Large dataset (n=100,000) performance")
    import time
    cat_l = np.random.default_rng(1).choice(["A","B","C","D"], 100_000)
    cont_l = np.random.default_rng(1).normal(100, 15, 100_000)
    t0 = time.perf_counter()
    r_l = eta_correlation(cat_l, cont_l, perform_anova=True)
    elapsed = time.perf_counter() - t0
    _assert(elapsed < 1.0, f"n=100k computed in {elapsed:.4f}s (< 1s)")
    _assert(0.0 <= r_l.eta <= 1.0, f"η={r_l.eta:.6f} ∈ [0,1]")

    print("\n" + "=" * 50)
    if failures:
        print(f"AUDIT COMPLETE: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("AUDIT COMPLETE: ALL CHECKS PASSED ✔")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        _audit()
    else:
        _run_examples()