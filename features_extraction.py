import os
import sys
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from transformers import TextClassificationPipeline
from nltk.tokenize import sent_tokenize


# arguments
# python feature_extraction.py data/input.csv data/features.csv models/roberta_euphemism models/roberta_sexism models/roberta_racism
INPUT_FILE_PATH = sys.argv[1]
OUTPUT_FILE_PATH = sys.argv[2]
CONTEXT_MODEL = sys.argv[3]
SEXISM_MODEL = sys.argv[4]
RACISM_MODEL = sys.argv[5]

def del_pet(data_col):
    new_col = [re.sub('\[(.*?)\]*\[(.*?)\]', 'ATTR', text) for text in data_col]
    return new_col

def del_marks(data_col):
    new_col = [re.sub('\[(.*?)\]*\[(.*?)\]', '', text) for text in data_col]
    return new_col

def get_target_text(data_col):
    new_col = []
    for text in data_col:
        target_text = ''
        for t in sent_tokenize(text):
            if "[PET_BOUNDARY]" in t:
                t = re.sub('\[(.*?)\]*\[(.*?)\]', 'ATTR', t)
                target_text += ' ' + t
        new_col.append(target_text)
            
    return new_col

class ClassificationLogits(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        return best_class

def get_features(data_col, model_id, features_dict, feature_title):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding='max_length', truncation=True, max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    pipe = ClassificationLogits(model=model, tokenizer=tokenizer)

    features_dict[feature_title +'_0']=[]
    features_dict[feature_title +'_1']=[]
    #features_dict[feature_title +'_2']=[]
    #features_dict[feature_title +'_3']=[]

    print('Logits extraction is started from ', model_id )
    for text in data_col:
        try:
            features_dict[feature_title +'_0'].append(float(pipe(text, batch_size=2, truncation="only_first")[0][0][0]))
            features_dict[feature_title +'_1'].append(float(pipe(text, batch_size=2, truncation="only_first")[0][0][1]))
            #features_dict[feature_title +'_2'].append(float(pipe(text, batch_size=2, truncation="only_first")[0][0][2]))
            #features_dict[feature_title +'_3'].append(float(pipe(text, batch_size=2, truncation="only_first")[0][0][3]))
        except:
            features_dict[feature_title +'_0'].append(0)
            features_dict[feature_title +'_1'].append(0)
            #features_dict[feature_title +'_2'].append(0)
            #features_dict[feature_title +'_3'].append(0)

df = pd.read_csv(INPUT_FILE_PATH)

text_column = get_target_text(df['text'])

features = {}

get_features(df['PET'], "models/roberta_PET/", features, 'PET')
get_features(text_column, CONTEXT_MODEL, features, 'context')
get_features(text_column, SEXISM_MODEL, features, 'sexism')
get_features(text_column, RACISM_MODEL, features, 'racism')
#get_features(del_marks(df['text']), SARCASM_MODEL, features, 'sarcasm')
#get_features(del_marks(df['text']), "cardiffnlp/twitter-xlm-roberta-base-sentiment", features, 'sentiment')


data = pd.DataFrame(features)
#data['label'] = df['label']
data['text'] = df['text']
data['PET'] = df['PET']
#df['sarcasm_0'] = data['sarcasm_0']
#df['sarcasm_1'] = data['sarcasm_1']
#df['sarcasm_2'] = data['sarcasm_2']
#df['sarcasm_3'] = data['sarcasm_3']


data.to_csv(OUTPUT_FILE_PATH, index=False)

