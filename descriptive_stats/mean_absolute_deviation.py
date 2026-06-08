import pandas as pd
import numpy as np

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Calculate Median - middle value of data
    median = np.median(data)

    # Calculate absolute deviations from median
    # i.e. how far each value is from the median
    absolute_deviations = np.abs(data - median)

    # Calculate MAD - median of all absolute deviations
    # more robust than std deviation against outliers
    mad = np.median(absolute_deviations)

    # Print results for current column
    print(f"{col}")
    print(f"  Median : {median:.4f}")
    print(f"  MAD    : {mad:.4f}")
    print()  # Empty line between columns for readability