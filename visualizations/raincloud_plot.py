import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    fig, ax = plt.subplots(figsize=(10, 5))

    # 1. CLOUD - half violin plot (top half only)
    sns.violinplot(y=data, ax=ax, color='steelblue',
                   inner=None, cut=0)

    # 2. BOX - box plot on top
    sns.boxplot(y=data, ax=ax, width=0.1,
                color='white',
                boxprops=dict(zorder=2),
                whiskerprops=dict(color='black'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker=''))

    # 3. RAIN - raw data points (sample 500 to avoid overplotting)
    sample = data.sample(n=min(500, len(data)), random_state=42)
    ax.scatter(np.random.uniform(-0.05, 0.05, size=len(sample)),
               sample, alpha=0.3, color='gray', s=10, zorder=3)

    # Add title and labels
    ax.set_title(f"Raincloud Plot of {col}", fontsize=14)
    ax.set_ylabel(col, fontsize=12)
    ax.set_xticks([])

    # Add grid for readability
    ax.grid(axis='y', alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/raincloud_{col}.png", dpi=150)
    plt.close()

    print(f"Raincloud Plot saved → outputs/raincloud_{col}.png")

print("\nAll raincloud plots done! ✅")