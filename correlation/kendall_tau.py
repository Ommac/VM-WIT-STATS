import pandas as pd
import numpy as np
from scipy import stats

# Read dataset
df = pd.read_csv("../data/kendall_tau.csv")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Loop through every pair of numeric columns
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):

        col1 = numeric_cols[i]
        col2 = numeric_cols[j]

        # Remove missing values
        combined = df[[col1, col2]].dropna()

        x = combined[col1].values
        y = combined[col2].values

        print(f"{col1}  vs  {col2}")

        # Skip constant columns
        if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
            print("  Cannot calculate Kendall Tau (constant column)")
            print()
            continue

        n = len(x)

        # Count Concordant and Discordant Pairs
        C = 0
        D = 0

        for a in range(n):
            for b in range(a + 1, n):

                dx = x[b] - x[a]
                dy = y[b] - y[a]

                if dx * dy > 0:
                    C += 1
                elif dx * dy < 0:
                    D += 1

        # Kendall Tau-B from scipy
        tau, p_value = stats.kendalltau(x, y)

        # Direction
        if tau > 0:
            direction = "Positive"
        elif tau < 0:
            direction = "Negative"
        else:
            direction = "No Direction"

        # Strength
        abs_tau = abs(tau)

        if abs_tau < 0.1:
            strength = "No Correlation"
        elif abs_tau < 0.3:
            strength = "Low Correlation"
        elif abs_tau < 0.5:
            strength = "Medium Correlation"
        elif abs_tau < 0.7:
            strength = "High Correlation"
        else:
            strength = "Very High Correlation"

        print(f"  Concordant Pairs : {C}")
        print(f"  Discordant Pairs : {D}")
        print(f"  Kendall Tau      : {tau:.4f}")
        print(f"  p-value          : {p_value:.4f}")
        print(f"  n                : {n}")
        print(f"  Direction        : {direction}")
        print(f"  Strength         : {strength}")

        if p_value < 0.05:
            print("  Significance     : Statistically Significant")
        else:
            print("  Significance     : Not Statistically Significant")

        print()