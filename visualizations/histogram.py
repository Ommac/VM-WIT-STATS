import pandas as pd
import matplotlib.pyplot as plt
import os

# Read dataset from CSV file
df = pd.read_csv("data/sample_data.csv")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Loop through every column in the dataset
for col in df.columns:

    # Select current column data
    data = df[col]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram with KDE curve
    ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    # Add title and labels
    ax.set_title(f"Histogram of {col}", fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/histogram_{col}.png", dpi=150)
    plt.close()

    print(f"Histogram saved → outputs/histogram_{col}.png")

print("\nAll histograms done! ✅")