import pandas as pd
import numpy as np
from scipy import stats

# Read dataset
df = pd.read_csv("data/sample_data.csv")

# Loop through all columns
for col in df.columns:
    data = df[col]
    mean = np.mean(data)
    std_error = stats.sem(data)

    confidence_interval = stats.t.interval(
        confidence=0.95,
        df=len(data)-1,
        loc=mean,
        scale=std_error
    )

    print(f"{col:20s} → 95% CI: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")