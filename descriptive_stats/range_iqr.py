import pandas as pd
import numpy as np

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Calculate Range - difference between max and min value
    data_range = np.max(data) - np.min(data)

    # Calculate Q1 - 25th percentile (lower quartile)
    Q1 = np.percentile(data, 25)

    # Calculate Q3 - 75th percentile (upper quartile)
    Q3 = np.percentile(data, 75)

    # Calculate IQR - middle 50% spread of data
    iqr = Q3 - Q1

    # Print results for current column
    print(f"{col}")
    print(f"  Range : {data_range:.4f}")
    print(f"  IQR   : {iqr:.4f}")
    print()  # Empty line between columns for readability