import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("../data/sample_data.csv")
# Get all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Loop through every pair of numeric columns
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = numeric_cols[i]
        col2 = numeric_cols[j]

        x = df[col1].dropna()
        y = df[col2].dropna()

        # Align both columns (same index)
        combined = pd.concat([x, y], axis=1).dropna()
        x = combined[col1]
        y = combined[col2]

        # Calculate Pearson correlation coefficient and p-value
        r, p_value = stats.pearsonr(x, y)

        # Calculate t-statistic : t = r * sqrt(n-2) / sqrt(1-r^2)
        n = len(x)
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)

        # Interpret strength of correlation
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "No correlation"
        elif abs_r < 0.3:
            strength = "Low correlation"
        elif abs_r < 0.5:
            strength = "Medium correlation"
        elif abs_r < 0.7:
            strength = "High correlation"
        else:
            strength = "Very High correlation"

        direction = "Positive" if r > 0 else "Negative"

        print(f"{col1}  vs  {col2}")
        print(f"  r         : {r:.4f}")
        print(f"  t-stat    : {t_stat:.4f}")
        print(f"  p-value   : {p_value:.4f}")
        print(f"  n         : {n}")
        print(f"  Result    : {direction} | {strength}")

        # Hypothesis test : H0: r=0, H1: r!=0
        if p_value < 0.05:
            print(f"  H0 rejected  → Significant correlation exists (p < 0.05)")
        else:
            print(f"  Fail to reject H0 → No significant correlation (p >= 0.05)")

        print()