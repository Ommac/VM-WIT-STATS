import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("../data/point_biserial.csv")

# Get numeric columns and binary columns (only 0 and 1 values)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

binary_cols = []
for col in numeric_cols:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        binary_cols.append(col)

continuous_cols = [col for col in numeric_cols if col not in binary_cols]

# Loop through each binary column vs each continuous column
for bin_col in binary_cols:
    for cont_col in continuous_cols:

        combined = pd.concat([df[bin_col], df[cont_col]], axis=1).dropna()
        binary     = combined[bin_col].values.astype(int)
        continuous = combined[cont_col].values

        print(f"{bin_col} (binary)  vs  {cont_col} (continuous)")

        # Skip constant columns
        if len(np.unique(binary)) <= 1 or len(np.unique(continuous)) <= 1:
            print("  Cannot calculate Point Biserial (constant column)")
            print()
            continue

        n  = len(continuous)
        n1 = np.sum(binary == 1)
        n0 = np.sum(binary == 0)

        mean1 = np.mean(continuous[binary == 1])   # mean of group 1
        mean0 = np.mean(continuous[binary == 0])   # mean of group 0
        Sx    = np.std(continuous, ddof=1)          # sample standard deviation

        # Check normality of continuous variable
        stat1, p_norm = stats.shapiro(continuous[:min(5000, len(continuous))])
        if p_norm < 0.05:
            print("  Warning: Continuous variable may not be normally distributed")

        # Formula : rpb = (mean1 - mean0) / Sx * sqrt(n1*n0 / n^2)
        rpb_formula = ((mean1 - mean0) / Sx) * np.sqrt((n1 * n0) / n**2)

        # scipy for p-value
        rpb, p_value = stats.pointbiserialr(binary, continuous)

        # Interpret strength
        abs_rpb = abs(rpb)
        if abs_rpb < 0.1:
            strength = "No Correlation"
        elif abs_rpb < 0.3:
            strength = "Low Correlation"
        elif abs_rpb < 0.5:
            strength = "Medium Correlation"
        elif abs_rpb < 0.7:
            strength = "High Correlation"
        else:
            strength = "Very High Correlation"

        direction = "Positive" if rpb > 0 else "Negative"

        print(f"  Group 1 mean   : {mean1:.4f}  (n={n1})")
        print(f"  Group 0 mean   : {mean0:.4f}  (n={n0})")
        print(f"  Sx             : {Sx:.4f}")
        print(f"  rpb (formula)  : {rpb_formula:.4f}")
        print(f"  rpb (scipy)    : {rpb:.4f}")
        print(f"  p-value        : {p_value:.4f}")
        print(f"  n              : {n}")
        print(f"  Direction      : {direction}")
        print(f"  Strength       : {strength}")

        # Hypothesis test : H0: rpb=0, H1: rpb!=0
        if p_value < 0.05:
            print(f"  Significance   : Statistically Significant")
        else:
            print(f"  Significance   : Not Statistically Significant")

        print()