import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("../data/spearman.csv")

# Get all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Loop through every pair of numeric columns
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):

        col1 = numeric_cols[i]
        col2 = numeric_cols[j]

        # Remove missing values
        combined = df[[col1, col2]].dropna()

        x = combined[col1]
        y = combined[col2]

        print(f"{col1}  vs  {col2}")

        # Skip constant columns
        if x.nunique() <= 1 or y.nunique() <= 1:
            print("  Cannot calculate Spearman Correlation (constant column)")
            print()
            continue

        n = len(x)

        # Spearman Rank Correlation
        rs, p_value = stats.spearmanr(x, y)

        # t-statistic
        if abs(rs) == 1:
            t_stat = float("inf")
        else:
            t_stat = rs * np.sqrt(n - 2) / np.sqrt(1 - rs**2)

        # Direction
        if rs > 0:
            direction = "Positive"
        elif rs < 0:
            direction = "Negative"
        else:
            direction = "No Direction"

        # Strength
        abs_rs = abs(rs)

        if abs_rs < 0.1:
            strength = "No Correlation"
        elif abs_rs < 0.3:
            strength = "Low Correlation"
        elif abs_rs < 0.5:
            strength = "Medium Correlation"
        elif abs_rs < 0.7:
            strength = "High Correlation"
        else:
            strength = "Very High Correlation"

        print(f"  Spearman Rho   : {rs:.4f}")
        print(f"  t-stat         : {t_stat:.4f}")
        print(f"  p-value        : {p_value:.4f}")
        print(f"  n              : {n}")
        print(f"  Direction      : {direction}")
        print(f"  Strength       : {strength}")

        # Hypothesis Test
        if p_value < 0.05:
            print("  Significance   : Statistically Significant")
        else:
            print("  Significance   : Not Statistically Significant")

        print()