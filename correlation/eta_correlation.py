import pandas as pd
import numpy as np
from scipy import stats

# Read dataset from CSV file
df = pd.read_csv("../data/eta_sample.csv")

# Get categorical columns (object type) and numeric columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()

# Loop through each categorical column vs each numeric column
for cat_col in categorical_cols:
    for num_col in numeric_cols:

        combined = pd.concat([df[cat_col], df[num_col]], axis=1).dropna()
        N = len(combined)

        print(f"{cat_col} (categorical)  vs  {num_col} (continuous)")

        # Skip constant columns
        if combined[cat_col].nunique() <= 1 or combined[num_col].nunique() <= 1:
            print("  Cannot calculate Eta (constant column)")
            print()
            continue

        # Grand mean of the continuous variable
        grand_mean = np.mean(combined[num_col])

        # Calculate SS_between and SS_within
        SS_between = 0
        SS_within  = 0

        for group_name, group_data in combined.groupby(cat_col)[num_col]:
            ni         = len(group_data)
            group_mean = np.mean(group_data)

            # Variation due to differences between groups
            SS_between += ni * (group_mean - grand_mean)**2

            # Variation due to differences within each group
            SS_within  += np.sum((group_data - group_mean)**2)

        SS_total = SS_between + SS_within

        # Eta formula : eta = sqrt(SS_between / SS_total)
        eta = np.sqrt(SS_between / SS_total)

        # t-statistic : t = sqrt( eta^2 * (N-2) / (1 - eta^2) )
        t_stat  = np.sqrt((eta**2 * (N - 2)) / (1 - eta**2))
        df_val  = N - combined[cat_col].nunique()
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_val))

        # Interpret strength
        if eta < 0.1:
            strength = "No Correlation"
        elif eta < 0.3:
            strength = "Low Correlation"
        elif eta < 0.5:
            strength = "Medium Correlation"
        elif eta < 0.7:
            strength = "High Correlation"
        else:
            strength = "Very High Correlation"

        print(f"  Grand mean     : {grand_mean:.4f}")
        print(f"  SS_between     : {SS_between:.4f}")
        print(f"  SS_within      : {SS_within:.4f}")
        print(f"  SS_total       : {SS_total:.4f}")
        print(f"  Eta            : {eta:.4f}")
        print(f"  t-stat         : {t_stat:.4f}")
        print(f"  p-value        : {p_value:.4f}")
        print(f"  N              : {N}")
        print(f"  Strength       : {strength}")

        # Hypothesis test : H0: eta=0, H1: eta!=0
        if p_value < 0.05:
            print(f"  Significance   : Statistically Significant")
        else:
            print(f"  Significance   : Not Statistically Significant")

        print()