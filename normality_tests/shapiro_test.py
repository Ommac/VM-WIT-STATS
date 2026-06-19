import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("../data/masterdataset.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Shapiro-Wilk test
    # returns test statistic (W) and p-value
    stat, p_value = stats.shapiro(data)

    # Interpret result based on p-value
    if p_value > 0.05:
        result = "NORMAL distribution ✅"
    else:
        result = "NOT normal distribution ❌"

    # Print results for current column
    print(f"{col}")
    print(f"  W statistic : {stat:.4f}")
    print(f"  p-value     : {p_value:.4f}")
    print(f"  Result      : {result}")
    print()