import pandas as pd
from glob import glob

from sklearn.ensemble import RandomForestClassifier


#python classification.py data/*_features.csv data/test.csv data/output.txt 
FEATURE_DIR = sys.argv[1]
TEST_FILE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3]


dfs = []
for file in glob(FEATURE_DIR):
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)

X = df.drop(columns=['text', 'PET', 'label'])
y = df['label']

clf = RandomForestClassifier(max_depth=1000, max_features=2, min_samples_leaf=10,
                             n_estimators=100, random_state=666)

clf.fit(X, y)

df_test = pd.read_csv(TEST_FILE_PATH)
X_test = df_test.drop(columns=['text','PET'])#,'context','sexism','racism'])


scores = clf.predict(X_test)

with open(OUTPUT_FILE_PATH, 'w') as file:
    for i in scores:
        file.write(str(i) + '\n')