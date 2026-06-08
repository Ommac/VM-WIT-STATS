import pandas as pd
from scipy.stats import skew, kurtosis

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Calculate Skewness - measures asymmetry of distribution
    # 0 = perfectly symmetric
    # positive = tail on right side
    # negative = tail on left side
    data_skew = skew(data)

    # Calculate Kurtosis - measures peakedness of distribution
    # 0 = normal distribution (mesokurtic)
    # positive = sharp peak (leptokurtic)
    # negative = flat peak (platykurtic)
    data_kurtosis = kurtosis(data)

    # Print results for current column
    print(f"{col}")
    print(f"  Skewness : {data_skew:.4f}")
    print(f"  Kurtosis : {data_kurtosis:.4f}")
    print()  # Empty line between columns for readability