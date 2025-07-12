"""
Script to create 2x2-plots showing the evaluation results, 
for the frequency and for the dispersion data. 
"""

# === Imports === 

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import join
import re


# === Parameters === 

## Select group of results 
#group = "frequency"
group = "dispersion_1000s-1f"
##group = "dispersion_1s-1000f"

# Don't change
groupindex = re.split("_", group)[0]

## Select plot file name 
#plotfilename = "2x2-plot_frequency-A.png"
#plotfilename = "2x2-plot_frequency-B.png"
#plotfilename = "2x2-plot_dispersion-A.png"
plotfilename = "2x2-plot_dispersion-B.png"


# Define the result files and corresponding rank columns; select 4 per plot! 
result_files = {
    #"chi_square": [join("results", group, "chi_square.csv"), "chi_square_rank"],
    #"LLR": [join("results", group, "LLR.csv"), "LLR_rank"],
    #"RRF": [join("results", group, "RRF.csv"), "RRF_rank"],
    #"tf_idf": [join("results", group, "tf_idf.csv"), "TF-IDF_rank"],
    "zeta": [join("results", group, "zeta_0.csv"), "zeta_rank"],
    "eta": [join("results", group, "eta.csv"), "eta_rank"],
    "ranksum": [join("results", group, "ranksum.csv"), "ranksum_rank"],
    "Welch": [join("results", group, "welch.csv"), "Welch_rank"],
}

# === Functions === 


# Function to read, sort, and rename rank column of the data
def process_file(filepath, rank_column, original_rank_col):
    df = pd.read_csv(filepath, sep="\t")
    df.sort_values(by=[groupindex], ascending=True, inplace=True)
    df.rename({original_rank_col: rank_column}, axis=1, inplace=True)
    return df

# Read and process all files with specific original rank column names
dfs = {
    #"chi_square": process_file(result_files["chi_square"][0], result_files["chi_square"][1], 'chi_square_rang'),
    #"LLR": process_file(result_files["LLR"][0], result_files["LLR"][1], 'LLR_rang'),
    #"RRF": process_file(result_files["RRF"][0], result_files["RRF"][1], 'rrf_dr0_rang'),
    #"tf_idf": process_file(result_files["tf_idf"][0], result_files["tf_idf"][1], 'tf_idf_rang'),
    "zeta": process_file(result_files["zeta"][0], result_files["zeta"][1], 'zeta_sd0_rang'),
    "eta": process_file(result_files["eta"][0], result_files["eta"][1], 'eta_sg0_rang'),
    "ranksum": process_file(result_files["ranksum"][0], result_files["ranksum"][1], 'ranksum_rang'),
    "Welch": process_file(result_files["Welch"][0], result_files["Welch"][1], 'welch_rang'),
}

# Set up the plot
mypalette = sns.light_palette("lime", as_cmap=False, n_colors=11)
fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=300)

# Plot each data set
plot_params = [
    #(dfs['chi_square'], 'chi_square_rank', axes[0, 0]),
    #(dfs['LLR'], 'LLR_rank', axes[0, 1]),
    #(dfs['RRF'], 'RRF_rank', axes[1, 0]),
    #(dfs['tf_idf'], 'TF-IDF_rank', axes[1, 1])
    (dfs['zeta'], 'zeta_rank', axes[0, 0]),
    (dfs['eta'], 'eta_rank', axes[0, 1]),
    (dfs['ranksum'], 'ranksum_rank', axes[1, 0]),
    (dfs['Welch'], 'Welch_rank', axes[1, 1])
]

for df, rank_col, ax in plot_params:

    # Sort the dataframe correctly 
    if groupindex == "dispersion": 
        df.sort_values(by="dispersion_numeric", ascending=True, inplace=True)

    # Define the data for the subplot
    sns.boxplot(
        x=groupindex, 
        y=rank_col, 
        data=df, 
        hue=groupindex, 
        legend=False, 
        palette=mypalette, 
        ax=ax,        
        )
    
    # Define the axes for the subplot
    ax.set_ylim(0.5, 150000)
    ax.set_yscale('log')
    ax.xaxis.grid(True)
    ax.tick_params(axis='x', rotation=45)
    # Set custom y-axis ticks and labels
    ax.set_yticks([0.5, 1, 10, 100, 1000, 10000, 100000, 150000])
    ax.set_yticklabels(['', '1', '10', '100', r'$10^3$', r'$10^4$', r'$10^5$', r''])

# Define the title etc. for the entire 2x2 plot
fig.suptitle(f'Ranks of the artificial word in the {groupindex} contrast analysis', fontsize=12, y=1.00)
plt.tight_layout()
#plt.show()
plt.savefig(join("visuals", plotfilename), dpi=300)
plt.close()
