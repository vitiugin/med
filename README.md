# Multilingual Euphemism Detection

This code is implementation of the model prepared for Multilingual Euphemism Detection Shared Task for the Fourth Workshop on Figurative Language Processing (FigLang 2024).


### 1. Run finetune.py script to finetune pre-trained language model:
```
python finetune.py username/target_euph models
```
where 
- username/target_euph - dataset
- models - path to folder for saving finetuned model


### 2. Run features_extraction.py script to create file with features for training or testing the model:
```
python features_extraction.py data/input.csv data/features.csv models/roberta_euphemism models/roberta_sexism models/roberta_racism
```
where
- data/input.csv - path to data file
- data/features.csv - path to output file
- models/roberta_euphemism, models/roberta_sexism, models/roberta_racism - path to pre-trained (finetuned) models


### 3. Run classification.py to create file with labels for test data:
```
python classification.py data/*_features.csv data/test.csv data/output.txt 
```
where
- data/*_features.csv - path to files with features for training model
- data/test.csv - path to file with test features 
- data/output.csv - path to file with labels