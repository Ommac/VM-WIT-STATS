import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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

    # Generate Q-Q plot
    # probplot returns (quantiles, values) and plots against normal distribution
    stats.probplot(data, dist="norm", plot=ax)

    # Add title
    ax.set_title(f"Q-Q Plot of {col}", fontsize=14)

    # Add grid for readability
    ax.grid(alpha=0.5)

    # Save plot to outputs folder
    plt.tight_layout()
    plt.savefig(f"outputs/qqplot_{col}.png", dpi=150)
    plt.close()

    print(f"Q-Q Plot saved → outputs/qqplot_{col}.png")

print("\nAll Q-Q plots done! ✅")