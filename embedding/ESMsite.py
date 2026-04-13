import pickle
import pandas as pd
from Bio import SeqIO
import numpy as np
import json
from tqdm import tqdm

# read esm embeddings
with open("./embedding/embedding_train_esm1v_t33_650M_UR90S_1.pkl", "rb") as f: # this file is very large, need to get by yourself.
    train_loaded_embeddings = pickle.load(f)
with open("./embedding/embedding_test_esm1v_t33_650M_UR90S_1.pkl", "rb") as f: # this file is very large, need to get by yourself.
    test_loaded_embeddings = pickle.load(f)

test_3gram_letter_3 = pd.read_csv('./data/Processed/Covid19/3gram/letter_1k_test.csv')
test_3gram_label_3 = pd.read_csv('./data/Processed/Covid19/3gram/label_1k_test.csv')
train_3gram_letter_3 = pd.read_csv('./data/Processed/Covid19/3gram/letter_1k_train.csv')
train_3gram_label_3 = pd.read_csv('./data/Processed/Covid19/3gram/label_1k_train.csv')
train_1280_label = train_3gram_label_3.copy()
test_1280_label = test_3gram_label_3.copy()

def get_last_non_na_column(row):
    non_na_columns = row.dropna().index
    return non_na_columns[-1] if len(non_na_columns) > 0 else None


input_path = './data/Processed/Covid19/ESM/key2keys_train.json' 
with open(input_path, 'r') as json_file:
    key2keys_train = json.load(json_file)
input_path = './data/Processed/Covid19/ESM/key2keys_test.json' 
with open(input_path, 'r') as json_file:
    key2keys_test = json.load(json_file)

embedding_key_to_matching_key = {
    value: key
    for key, values in key2keys_test.items()
    for value in values
}

count_find = 0
count_missing = 0

positions = test_1280_label['Position']
position_indices = positions.str.split('|').str[1].astype(int) - 1  
last_non_nan_indices = test_1280_label.apply(lambda row: test_1280_label.columns.get_loc(get_last_non_na_column(row)), axis=1)

for idx, row in tqdm(test_1280_label.iterrows(), total=len(test_1280_label), desc="Processing rows"):
    position = row['Position']
    position_index = position_indices[idx]
    last_non_nan_index = last_non_nan_indices[idx]

    for col in test_1280_label.columns[3:last_non_nan_index+1]:
        embedding_key = f"{position}.{col}"
        if pd.isna(test_1280_label.at[idx, col]):
            continue
        else:
        
            if embedding_key in embedding_key_to_matching_key:
                matching_key = embedding_key_to_matching_key[embedding_key]
                embedding_list = test_loaded_embeddings[matching_key][position_index].tolist()
                test_1280_label.at[idx, col] = embedding_list
                count_find += 1
            else:
                print(f"Missing embedding for {embedding_key}")
                count_missing += 1

# 输出结果
print(f"Total embeddings found: {count_find}")
print(f"Total embeddings missing: {count_missing}") # if there are missing embeddings, we need to check the key2keys_test.json file and the test_1280_label file to see if there are any mismatches in the keys.


embedding_key_to_matching_key = {
    value: key
    for key, values in key2keys_train.items()
    for value in values
}

count_find = 0
count_missing = 0

positions = train_1280_label['Position']
position_indices = positions.str.split('|').str[1].astype(int) - 1  
last_non_nan_indices = train_1280_label.apply(lambda row: train_1280_label.columns.get_loc(get_last_non_na_column(row)), axis=1)

for idx, row in tqdm(train_1280_label.iterrows(), total=len(train_1280_label), desc="Processing rows"):
    position = row['Position']
    position_index = position_indices[idx]
    last_non_nan_index = last_non_nan_indices[idx]

    for col in train_1280_label.columns[3:last_non_nan_index+1]:
        embedding_key = f"{position}.{col}"
        if pd.isna(train_1280_label.at[idx, col]):
            continue
        else:
        
            if embedding_key in embedding_key_to_matching_key:
                matching_key = embedding_key_to_matching_key[embedding_key]
                embedding_list = train_loaded_embeddings[matching_key][position_index].tolist()
                train_1280_label.at[idx, col] = embedding_list
                count_find += 1
            else:
                print(f"Missing embedding for {embedding_key}")
                count_missing += 1

print(f"Total embeddings found: {count_find}")
print(f"Total embeddings missing: {count_missing}")

test_1280_letter = test_1280_label.copy()
test_1280_letter['Label'] = test_3gram_letter_3['Label']
test_1280_label.to_csv('./data/Processed/Covid19/ESM/esm1v1_label_test.csv', index=False)
test_1280_letter.to_csv('./data/Processed/Covid19/ESM/esm1v1_letter_test.csv', index=False)
train_1280_letter = train_1280_label.copy()
train_1280_letter['Label'] = train_3gram_letter_3['Label']
train_1280_label.to_csv('./data/Processed/Covid19/ESM/esm1v1_label_train.csv', index=False)
train_1280_letter.to_csv('./data/Processed/Covid19/ESM/esm1v1_letter_train.csv', index=False)

# columns = ['Position', 'predict_date', 'y',0,1,2,3,4]

# train_1280_letter_last5 = pd.DataFrame(columns=columns)
# train_1280_letter_last5['Position'] = train_1280_letter['Position'].copy()
# train_1280_letter_last5['predict_date'] = train_1280_letter['predict_date'].copy()
# train_1280_letter_last5['y'] = train_1280_letter['Label'].copy()

# train_1280_letter_last5.iloc[:, 3:] = pd.DataFrame(train_1280_letter.iloc[:,3:].apply(lambda row: 
#                                                                 row.dropna().iloc[-5:].values, axis=1).tolist())
# test_1280_letter_last5 = pd.DataFrame(columns=columns)
# test_1280_letter_last5['Position'] = test_1280_letter['Position'].copy()
# test_1280_letter_last5['predict_date'] = test_1280_letter['predict_date'].copy()
# test_1280_letter_last5['y'] = test_1280_letter['Label'].copy()


# test_1280_letter_last5.iloc[:, 3:] = pd.DataFrame(test_1280_letter.iloc[:,3:].apply(lambda row:
#                                                                 row.dropna().iloc[-5:].values, axis=1).tolist())

# columns = ['Position', 'predict_date', 'y',0,1,2,3,4]

# train_1280_label_last5 = pd.DataFrame(columns=columns)
# train_1280_label_last5['Position'] = train_1280_label['Position'].copy()
# train_1280_label_last5['predict_date'] = train_1280_label['predict_date'].copy()
# train_1280_label_last5['y'] = train_1280_label['Label'].copy()

# train_1280_label_last5.iloc[:, 3:] = pd.DataFrame(train_1280_label.iloc[:,3:].apply(lambda row: 
#                                                                 row.dropna().iloc[-5:].values, axis=1).tolist())
# test_1280_label_last5 = pd.DataFrame(columns=columns)
# test_1280_label_last5['Position'] = test_1280_label['Position'].copy()
# test_1280_label_last5['predict_date'] = test_1280_label['predict_date'].copy()
# test_1280_label_last5['y'] = test_1280_label['Label'].copy()


# test_1280_label_last5.iloc[:, 3:] = pd.DataFrame(test_1280_label.iloc[:,3:].apply(lambda row:
#                                                                 row.dropna().iloc[-5:].values, axis=1).tolist())

# test_1280_label_last5.iloc[:,2:].to_csv('./data/Processed/Covid19/ESM/esm1v1_label_last5_test.csv', index=False)
# test_1280_letter_last5.iloc[:,2:].to_csv('./data/Processed/Covid19/ESM/esm1v1_letter_last5_test.csv', index=False)
# train_1280_label_last5.iloc[:,2:].to_csv('./data/Processed/Covid19/ESM/esm1v1_label_last5_train.csv', index=False)
# train_1280_letter_last5.iloc[:,2:].to_csv('./data/Processed/Covid19/ESM/esm1v1_letter_last5_train.csv', index=False)