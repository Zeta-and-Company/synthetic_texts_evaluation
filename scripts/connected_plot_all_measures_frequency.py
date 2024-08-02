#@author JuliaDudar

import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# List of measures to iterate over
measures = [
    'tf_idf', 'zeta_sd0', 'zeta_sd2', 'rrf_dr0', 'eta_sg0',
    'welsh', 'ranksum', 'chi_square', 'LLR'
]

# Dictionary to map original rank column names to new names
rank_column_rename = {
    'zeta_sd2_rank': 'zeta_log_rank',
    'zeta_sd0_rank': 'zeta_rank',
    'rrf_dr0_rank': 'RRF_rank',
    'eta_sg0_rank': 'eta_rank',
    'welsh_rank': 'welch_rank'
}

for measure in measures:
    # Update file paths according to the measure
    all_files = glob.glob(f"D:/synthetic_experiments/output_data/frequency_manipulation_results/synthetic_{measure}_results/*.csv")
    resultfile = f"D:/synthetic_experiments/output_data/frequency_manipulation_results/{measure}_connected_plot_x.csv"

    added_data = pd.DataFrame()
    for filename in all_files:
        df = pd.read_csv(filename, sep="\t")
        added_data = added_data.append(df, ignore_index=True, sort=False)
        added_data.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')

    # Ensure the directory for results file exists
    os.makedirs(os.path.dirname(resultfile), exist_ok=True)

    with open(resultfile, "w", encoding='utf-8') as outfile:
        added_data.to_csv(outfile, sep="\t")

    df = pd.read_csv(resultfile, sep="\t")
    df.sort_values(by=['frequency'], ascending=True, inplace=True)

    # Rename the rank column if it exists in the dataframe
    rank_col = f"{measure}_rank"
    if rank_col in df.columns:
        df.rename(columns={rank_col: rank_column_rename.get(rank_col, rank_col)}, inplace=True)

    plot_rank_col = rank_column_rename.get(rank_col, rank_col)
    if plot_rank_col not in df.columns:
        print(f"Column {plot_rank_col} not found in data for measure {measure}")
        continue

    ax = plt.axes()
    sns.boxplot(x=df['frequency'], y=df[plot_rank_col], data=df, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(y=df[plot_rank_col], x=df['frequency'], data=df, size=1, color=".3", linewidth=0)

    ax.xaxis.grid(True)
    ax.set_ylim(1, 100000)
    ax.set_yscale('log')
    plt.title('Ranks of the Artificial Word According to Frequency Variation Analysis')
    plt.show()
