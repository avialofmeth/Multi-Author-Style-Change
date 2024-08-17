# Multi-author-style-change
Group members: 
Yuwei Shen, Qing Li and Shiling Deng

## Overview
This project is the final exam assignment of the course Language Processing 2 at the University of Copenhagen. In this code, we employed three approaches: Prompt engineering, BERT and TF-IDF (Baseline).


## Dataset
The PAN 24 Style Change Detection dataset can be downloaded here: https://zenodo.org/records/10677876. Run `preprocess.ipynb` to preprocess the data.

## Usage
### Prompt Engineering
Run `prompting.ipynb` to get LLM generated responses; Run `process_json.ipynb` to deal with some responses with incorrect format; Run `evaluation.ipynb` to get evaluation results. 

### BERT
For fine-tuning, refer to `fine_tune_bert.ipynb`; for random forest classifier, refer to `train_test_random_forest_classifier.ipynb`; for attention visualization, refer to `visualize_attention.ipynb`. 

### Linguistic method

For the baseline model, refer to `linguistic_method_baseline.ipynb`.

# Results
Table 1: F1 scores on our test set of the Medium hard task  

| Task| Prompt Engineering|BERT + Random Forest|Fine-tuned BERT + Random Forest|Linguistic method (baseline)
| ----------- | ----------- | ----------- | ----------- | ----------- 
| Medium       | 0.594       | 0.795       | 0.806       | 0.811      

For more details, please refer to the **final report** PDF.  

