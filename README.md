# synthetic_texts_evaluation

## Data and code for CCLS2025 paper 

This repository provides the data and code for the following preprint: Julia Havrylash and Christof Schöch (2025). "Exploring Measures of Distinctiveness. An Evaluation Using Synthetic Texts". In: _CCLS2025 Conference Preprints_ 4 (1). DOI: [10.26083/tuprints-00030152](https://doi.org/10.26083/tuprints-00030152). 

## How to replicate our findings 

1. Download or clone this repository 
1. Create a virtual environment for the Python scripts, installing the modules listed in `requirements.txt`. 
1. The corpus of synthetic texts is available in the folder `corpus`. (The underlying full text novels are not available here, due to copyright restrictions). Therefore, the script `1_synthetic_texts_generation.py` cannot be run. 
1. Run pydistinto. 
1. To create the 2x2-visualizations from the paper, run `4_make-2x2-plots.py`. It takes the dataframes with the detailed results and creates a plot showing the relationship between the frequency contrast and the rank of the artificial word. Each plot gathers results for four measures. You need to run this script 4 times with varying measures and conditions to obtain all visuals. 


