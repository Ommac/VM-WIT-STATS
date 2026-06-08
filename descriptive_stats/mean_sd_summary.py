import pandas as pd
import numpy as np

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Calculate Mean - average of all values
    mean = np.mean(data)

    # Calculate Standard Deviation - spread of data
    std = np.std(data)

    # Calculate lower bound - one SD below mean
    lower = mean - std

    # Calculate upper bound - one SD above mean
    upper = mean + std

    # Print results for current column
    print(f"{col}")
    print(f"  Mean     : {mean:.4f}")
    print(f"  SD       : {std:.4f}")
    print(f"  Mean±SD  : ({lower:.4f}  to  {upper:.4f})")
    print()  # Empty line between columns for readability