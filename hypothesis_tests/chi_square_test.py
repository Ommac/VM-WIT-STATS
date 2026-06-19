import pandas as pd
from scipy import stats

# Read dataset
df = pd.read_csv("../data/chi_square_test.csv")

# First two columns are categorical
col1 = df.columns[0]
col2 = df.columns[1]

# Create contingency table
table = pd.crosstab(df[col1], df[col2])

# Chi-Square Test
chi2, p_value, dof, expected = stats.chi2_contingency(table)

print(f"{col1} vs {col2}")
print("\nObserved Frequency Table")
print(table)

print("\nExpected Frequency Table")
print(expected)

print(f"\nChi-Square Statistic : {chi2:.4f}")
print(f"Degrees of Freedom   : {dof}")
print(f"p-value              : {p_value:.4f}")

print("\nHypothesis")
print("H0 : Variables are independent")
print("H1 : Variables are associated")

if p_value < 0.05:
    print("\nResult      : Reject H0")
    print("Conclusion  : Significant Association Exists")
else:
    print("\nResult      : Fail to Reject H0")
    print("Conclusion  : No Significant Association")