import pandas as pd
import numpy as np

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Calculate Variance - average of squared differences from mean
    # measures how spread out the data is
    variance = np.var(data)

    # Calculate Standard Deviation - square root of variance
    # same unit as original data, easier to interpret
    std_deviation = np.std(data)

    # Print results for current column
    print(f"{col}")
    print(f"  Variance  : {variance:.4f}")
    print(f"  Std Dev   : {std_deviation:.4f}")
    print()  # Empty line between columns for readability