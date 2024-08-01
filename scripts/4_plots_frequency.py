import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Define the result files and corresponding rank columns
result_files = {
    "chi_square": ("D:/synthetic_experiments/output_data/frequency_manipulation_results/chi_square_connected_plot.csv", "chi_square_rank"),
    "LLR": ("D:/synthetic_experiments/output_data/frequency_manipulation_results/LLR_connected_plot.csv", "LLR_rank"),
    "RRF": ("D:/synthetic_experiments/output_data/frequency_manipulation_results/RRF_connected_plot.csv", "RRF_rank"),
    "tf_idf": ("D:/synthetic_experiments/output_data/frequency_manipulation_results/tf_idf_connected_plot.csv", "TF-IDF_rank")
}

# Function to read, sort, and rename rank column of the data
def process_file(filepath, rank_column, original_rank_col):
    df = pd.read_csv(filepath, sep="\t")
    df.sort_values(by=['frequency'], ascending=True, inplace=True)
    df.rename({original_rank_col: rank_column}, axis=1, inplace=True)
    return df

# Read and process all files with specific original rank column names
dfs = {
    "chi_square": process_file(result_files["chi_square"][0], result_files["chi_square"][1], 'chi_square_rang'),
    "LLR": process_file(result_files["LLR"][0], result_files["LLR"][1], 'LLR_rang'),
    "RRF": process_file(result_files["RRF"][0], result_files["RRF"][1], 'rrf_dr0_rang'),
    "tf_idf": process_file(result_files["tf_idf"][0], result_files["tf_idf"][1], 'tf_idf_rang')
}

# Set up the plot
bright_green_palette = sns.light_palette("lime", as_cmap=False)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)

# Plot each data set
plot_params = [
    (dfs['chi_square'], 'chi_square_rank', axes[0, 0]),
    (dfs['LLR'], 'LLR_rank', axes[0, 1]),
    (dfs['RRF'], 'RRF_rank', axes[1, 0]),
    (dfs['tf_idf'], 'TF-IDF_rank', axes[1, 1])
]

for df, rank_col, ax in plot_params:
    sns.boxplot(x='frequency', y=rank_col, data=df, palette=bright_green_palette, ax=ax)
    ax.set_ylim(1, 10000)
    ax.set_yscale('log')
    ax.xaxis.grid(True)

fig.suptitle('Ranks of the Artificial Word According to Frequency Variation Analysis', fontsize=12, y=1.00)
plt.tight_layout()
plt.show()
