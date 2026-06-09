import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Plot violin plot using seaborn
    # inner="box" shows box plot inside violin
    sns.violinplot(y=data, ax=ax, color='steelblue', inner='box')

    # Add title and labels
    ax.set_title(f"Violin Plot of {col}", fontsize=14)
    ax.set_ylabel(col, fontsize=12)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/violinplot_{col}.png", dpi=150)
    plt.close()

    print(f"Violin Plot saved → outputs/violinplot_{col}.png")

print("\nAll violin plots done! ✅")