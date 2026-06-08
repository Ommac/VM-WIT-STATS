import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Perform Anderson-Darling test
    # returns statistic, critical values, significance levels
    result = stats.anderson(data, dist='norm')

    # Get critical value at 5% significance level (index 2)
    critical_value_5 = result.critical_values[2]

    # Interpret result
    if result.statistic < critical_value_5:
        conclusion = "NORMAL distribution ✅"
    else:
        conclusion = "NOT normal distribution ❌"

    # Print results for current column
    print(f"{col}")
    print(f"  A² statistic    : {result.statistic:.4f}")
    print(f"  Critical Value  : {critical_value_5:.4f} (at 5%)")
    print(f"  Result          : {conclusion}")
    print()