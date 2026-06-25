import pandas as pd
import matplotlib.pyplot as plt
import os

# Read dataset from CSV file
df = pd.read_csv("data/support2.csv")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot boxplot
    # patch_artist fills the box with color
    # notch shows confidence interval around median
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor='steelblue', color='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', color='gray', alpha=0.3))

    # Add title and labels
    ax.set_title(f"Box Plot of {col}", fontsize=14)
    ax.set_ylabel(col, fontsize=12)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{col}.png", dpi=150)
    plt.close()

    print(f"Box Plot saved → outputs/boxplot_{col}.png")

print("\nAll box plots done! ✅")