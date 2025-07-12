# synthetic_texts_evaluation

## Data and code for CCLS2025 paper 

This repository provides the data and code for the following preprint: Julia Havrylash and Christof Schöch (2025). "Exploring Measures of Distinctiveness. An Evaluation Using Synthetic Texts". In: _CCLS2025 Conference Preprints_ 4 (1). DOI: [10.26083/tuprints-00030152](https://doi.org/10.26083/tuprints-00030152). 


## What you find in this repository 

- The folder `corpus` contains the complete collection of 320 synthetic texts (with word form, lemma and pos, but consisting of tokens sampled randomly from the entire base corpus). The underlying full text novels are not available here, due to copyright restrictions. Therefore, the script `1_synthetic_texts_generation.py` cannot be run. However, a metadata table is available, showing the composition of the underlying corpus. 
- The folder `results` contains a number of tables documenting the rank of the target word in each run. This represents the raw results of the analysis. 
- The folder `scripts` contains, wait for it, all the Python scripts used for this work, including a copy of the pydistinto code used here. 
- The folder `visuals` contains the figures as they can also be found in the paper. These figures are all produced by the script `4_make-2x2-plots.py`. 

## How to replicate our findings 

1. Download or clone this repository 
1. Create a virtual environment for the Python scripts, installing the modules listed in `requirements.txt`. 
1. To just recreate the 2x2-visualizations from the paper, run `4_make-2x2-plots.py`. It takes the dataframes with the detailed results and creates a plot showing the relationship between the frequency contrast and the rank of the artificial word. Each plot gathers results for four measures. You need to run this script 4 times with varying measures and conditions to obtain all visuals. 
1. To go further back and run the analyses producing the raw data, you need to use pydistinto
1. To go even further back and generate synthetic texts based on the original novels, rather than accepting these as given, you would need to obtain access to the in-copyright full texts, which is currently not available. However,


