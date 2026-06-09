import pandas as pd
import matplotlib.pyplot as plt
import os

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Create Age groups (bins of 10 years)
df['Age_Group'] = pd.cut(df['Age'], bins=range(0, 91, 10),
                          labels=['0-10','10-20','20-30','30-40',
                                  '40-50','50-60','60-70','70-80','80-90'])

# Get all columns except Age and Age_Group
numeric_cols = [col for col in df.columns if col not in ['Age', 'Age_Group']]

# Loop through every numeric column
for col in numeric_cols:

    # Calculate mean of column for each age group
    grouped = df.groupby('Age_Group', observed=True)[col].mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot line chart
    ax.plot(grouped.index, grouped.values,
            marker='o', color='steelblue',
            linewidth=2, markersize=6)

    # Add title and labels
    ax.set_title(f"Mean {col} by Age Group", fontsize=14)
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel(f"Mean {col}", fontsize=12)

    # Add grid for readability
    ax.grid(alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/linechart_{col}.png", dpi=150)
    plt.close()

    print(f"Line Chart saved → outputs/linechart_{col}.png")

print("\nAll line charts done! ✅")