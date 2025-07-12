"""
Script.
"""

import glob
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

# List of parameters to iterate over
parameters = [
    "1000/1", "500/2", "200/5", "100/10", "50/20",
    "20/50", "10/100", "5/200", "2/500", "1/1000"
]
# List of measures to iterate over
measures = [
    'tf_idf', 'zeta_sd0', 'zeta_sd2', 'rrf_dr0', 'eta_sg0',
    'welsh', 'ranksum', 'chi_square', 'LLR'
]

#for the results with 1000s_1f in comparison corpus, just change 1s_1000f to 1000s_1f

for param in parameters:
    # Replace slash with underscore for file paths
    param_underscore = param.replace("/", "_")
    # Update file paths according to the parameter and measure
    all_files = glob.glob(f"D:/synthetic_experiments/output_data/results_dispersion_{param_underscore}_1s_1000f/*.csv")
    resultsfile = f"D:/synthetic_experiments/output_data/dispersion_manipulation_results/generated_data_results_dispersion_manipulation/gen_results_dispersion_{param_underscore}_1s_1000f.csv"

    sorted_data = {}
    added_data = pd.DataFrame()

    for filename in all_files:
        df = pd.read_csv(filename, sep="\t")
        df.drop(['docprops1', 'docprops2', 'meanrelfreqs', 'relfreqs1', 'relfreqs2'], axis=1, inplace=True)

        for column in df:
            df = df.sort_values(by=column, ascending=False).reset_index(drop=True)
            column = pd.Series(data=df[column], index=df.index)
            words = pd.Series(data=df.iloc[:, 0], index=df.index)
            rank = pd.Series(range(1, 1 + len(df)))
            sorted_data.update({column.name: column, column.name + "_words": words, column.name + "_rank": rank})

        data = pd.DataFrame.from_dict(sorted_data)
        data.drop(['Unnamed: 0', 'Unnamed: 0_words', 'Unnamed: 0_rank'], axis=1, inplace=True)
        final_data = data.loc[:, :]
        added_data = added_data.append(final_data, ignore_index=True)
    for measure in measures:
        try:
            synthetic_word_file = f'D:/synthetic_experiments/output_data/dispersion_manipulation_results/{measure}_generated_data_1s_1000f/{measure}_{param_underscore}_1s_1000f.csv'
            measure_words_col = f'{measure}_words'
            measure_rank_col = f'{measure}_rank'

            if measure_words_col not in added_data.columns or measure_rank_col not in added_data.columns:
                print(f"Columns {measure_words_col} or {measure_rank_col} not found in data for parameter {param}")
                continue

            searched_word = added_data.loc[added_data[measure_words_col] == 'untuniutntrng55886']
            searched_word.reset_index(inplace=True)
            print(len(searched_word))

            plot = searched_word[[measure_words_col, measure, measure_rank_col]]
            plot.insert(3, 'dispersion', param)
            plot.insert(4, 'dispersion_numeric', eval(param.replace("/", "/")))
            os.makedirs(os.path.dirname(synthetic_word_file), exist_ok=True)
            with open(synthetic_word_file, "w", encoding='utf-8') as outfile:
                plot.to_csv(outfile, sep="\t")

            print(plot)
            #sns.stripplot(data=plot, x=plot[measure_rank_col], y=plot[measure])
            #plt.show()
        except KeyError as e:
            print(f"KeyError: {e} for measure {measure} and parameter {param}")
    with open(resultsfile, "w", encoding='utf-8') as outfile:
       added_data.to_csv(outfile, sep="\t")
