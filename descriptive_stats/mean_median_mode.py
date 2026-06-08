import pandas as pd
import numpy as np

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:
    
    # Select current column data
    data = df[col]

    # Calculate Mean - sum of all values divided by count
    mean = np.mean(data)

    # Calculate Median - middle value when data is sorted
    median = np.median(data)

    # Calculate Mode - most frequently occurring value
    mode = data.mode()[0]

    # Print results for current column
    print(f"{col}")
    print(f"  Mean   : {mean:.4f}")
    print(f"  Median : {median:.4f}")
    print(f"  Mode   : {mode:.4f}")
    print()  # Empty line between columns for readability