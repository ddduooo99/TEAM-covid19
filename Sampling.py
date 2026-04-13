from Bio import Phylo 
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
from random import sample
import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict

def load_tree(nwk_file):
    return Phylo.read(nwk_file, "newick")

def dfs_paths(node, current_path, all_paths, tree):
    current_path.append(node)
    if node not in tree or len(tree[node]) == 0:
        all_paths.append(tuple(current_path))  
    else:
        for child in tree[node]:
            dfs_paths(child, current_path, all_paths, tree)
    current_path.pop() 

def find_all_unique_paths(tree):
    all_paths = []
    root_lineages = [lineage for lineage in tree if '.' not in lineage]  
    for root in root_lineages:
        dfs_paths(root, [], all_paths, tree)
    return set(all_paths)  

def update_description(data):
    if data['Lineage'].startswith("A.") or data['Lineage'].startswith("B.") or data['Lineage'] == "A" or data['Lineage'] == "B":
        return data['Lineage']
    elif data['Description'].startswith("Alias of B."):
        b_value = data['Description'].split(",")[0].replace("Alias of ", "")
        if len(b_value.split(" ")) > 1 and b_value.split(" ")[0].startswith("B."):
            return b_value.split(" ")[0]
        else:   
            return b_value
    else:
        return data['Description']
    
def extract_parent(description):
    if description in ["A", "B"]:
        return description
    elif '.' in description:
        return description.rsplit('.', 1)[0]  
    else:
        return description



def get_genbank_datasets(N=500, pathlen=6):

    lineage_notes = pd.read_csv("./data/supplyment/lineage_notes.txt", sep="\t")
    lineage_notes['Description'] = lineage_notes.apply(lambda row: update_description(row), axis=1)
    
    lineage_notes_filtered = lineage_notes[~lineage_notes.iloc[:, 0].str.startswith("X")]
    lineage_notes_filtered = lineage_notes_filtered[~lineage_notes_filtered.iloc[:, 1].str.startswith("Alias of X")]
    lineage_notes_filtered = lineage_notes_filtered[~lineage_notes_filtered.iloc[:, 0].str.startswith("*")]
    lineage_notes_filtered['parent'] = lineage_notes_filtered['Description'].apply(lambda x: extract_parent(x))

    metadata = pd.read_csv('./data/supplyment/public-latest.metadata.tsv', sep='\t')
    metadata = metadata.drop(columns=['completeness'])
    metadata = metadata.iloc[:, :-2]

    metadata_human = metadata[metadata['host'] == 'Homo sapiens'].copy()
    metadata_human = metadata_human.drop(columns=['host'])
    metadata_human = metadata_human[~metadata_human["pangolin_lineage"].isna()]

    lineage_map = dict(zip(lineage_notes_filtered.iloc[:, 0], lineage_notes_filtered.iloc[:, 1]))
    metadata_human['pangolin_lineage'] = metadata_human['pangolin_lineage'].map(lineage_map)
    metadata_human = metadata_human[~metadata_human["pangolin_lineage"].isna()]
    
    id_list = pd.read_csv('./data/ID.txt', header=None)[0].tolist()
    metadata_human = metadata_human[metadata_human['genbank_accession'].isin(id_list)]

    unique_lineages_select = set(metadata_human['pangolin_lineage']) 
    lineage_tree_select = defaultdict(list)
    for lineage in unique_lineages_select:
        parent_lineage = '.'.join(lineage.split('.')[:-1]) if '.' in lineage else None
        if parent_lineage:
            lineage_tree_select[parent_lineage].append(lineage)

    unique_paths_select = find_all_unique_paths(lineage_tree_select)

    metadata_human_select = metadata_human[metadata_human['date'] <= '2022-04-01'].copy()
    metadata_human_select = metadata_human_select[~metadata_human_select['genbank_accession'].isna()]
    metadata_human_select = metadata_human_select[metadata_human_select['date'].str.contains(r'-\d{2}-', na=False)]
    metadata_human_select.loc[:, 'season'] = pd.to_datetime(metadata_human_select['date']).dt.to_period('Q').astype(str)

    unique_paths_select_min = [path for path in unique_paths_select if len(path) >= pathlen]
    unique_paths_select_pathlen = []
    for path in unique_paths_select_min:
        if len(path) > pathlen:
            for i in range(len(path) - pathlen + 1): 
                unique_paths_select_pathlen.append(path[i:i + pathlen])
        else:
            unique_paths_select_pathlen.append(path)

    indexed_paths = {index: path for index, path in enumerate(unique_paths_select_pathlen)}

    grouped_metadata = metadata_human_select.sort_values(by='date').groupby('pangolin_lineage')
    
    accessions_rows = []
    dates_rows = []

    for index, path in tqdm(indexed_paths.items(), desc="Processing lineages"):
        for i in range(N):
            datenow = metadata_human_select['date'].min()
            sampled_genbank = []
            sampled_date = []
            
            for count, lineage in enumerate(path):
                if lineage not in grouped_metadata.groups:
                    sampled_genbank.append(None); sampled_date.append(None)
                    continue

                lineage_group = grouped_metadata.get_group(lineage)
                metadata_selected = lineage_group[lineage_group['date'] >= datenow]

                if metadata_selected.empty:
                    sampled_genbank.append(None); sampled_date.append(None)
                else:
                    sampled_row = metadata_selected.sample(n=1)
                    datenow = sampled_row['date'].values[0]
                    sampled_date.append(datenow)
                    sampled_genbank.append(sampled_row['genbank_accession'].values[0])

            accessions_rows.append([index] + sampled_genbank)
            dates_rows.append([index] + sampled_date)

    cols = ['indexed_path'] + list(range(pathlen))
    genbank_accessions = pd.DataFrame(accessions_rows, columns=cols)
    genbank_date = pd.DataFrame(dates_rows, columns=cols)

    genbank_accessions = genbank_accessions.dropna().drop_duplicates()
    genbank_date = genbank_date.loc[genbank_accessions.index]
    genbank_accessions = genbank_accessions.reset_index(drop=True)
    genbank_date = genbank_date.reset_index(drop=True)

    genbank_date_season = genbank_date.copy()
    for col in genbank_date_season.columns[1:]: 
        genbank_date_season[col] = pd.to_datetime(genbank_date_season[col], errors='coerce').dt.to_period('Q').astype(str)

    genbank_lineage = genbank_accessions.copy()
    accession_to_lineage = metadata_human_select.set_index('genbank_accession')['pangolin_lineage'].to_dict()

    for row_index in range(len(genbank_lineage)):
        for col_index in range(1, len(genbank_lineage.columns)):
            accession = genbank_lineage.iloc[row_index, col_index]
            genbank_lineage.iloc[row_index, col_index] = accession_to_lineage.get(accession, None)

    time_range = ["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4","2022Q1","2022Q2"]
    genbank_accessions_season = pd.DataFrame(index=range(len(genbank_accessions)), columns=['indexed_path'] + time_range)
    genbank_lineage_season = pd.DataFrame(index=range(len(genbank_accessions)), columns=['indexed_path'] + time_range)
    genbank_accessions_season['indexed_path'] = genbank_accessions['indexed_path']
    genbank_lineage_season['indexed_path'] = genbank_accessions['indexed_path']

    for row_index in range(len(genbank_date_season)):
        for col_index in range(1, len(genbank_date_season.columns)):  
            season = genbank_date_season.iloc[row_index, col_index]  
            if season in time_range:
                genbank_accessions_season.loc[row_index, season] = genbank_accessions.iloc[row_index, col_index] 
                genbank_lineage_season.loc[row_index, season] = genbank_lineage.iloc[row_index, col_index] 

    genbank_lineage_season_filled = genbank_lineage_season.copy()
    genbank_accessions_season_filled = genbank_accessions_season.copy()
    lineages = metadata_human_select['pangolin_lineage'].unique()
    
    columns_season = genbank_accessions_season_filled.columns[1:]

    for row_index in tqdm(range(len(genbank_accessions_season_filled)), desc="Filling NA"):
        for col_idx, season in enumerate(columns_season):
            if pd.isna(genbank_accessions_season_filled.at[row_index, season]):
                row_lineages = genbank_lineage_season_filled.iloc[row_index, 1:].values
                
                prev_non_empty = next((row_lineages[i] for i in range(col_idx - 1, -1, -1) if pd.notna(row_lineages[i])), None)
                next_non_empty = next((row_lineages[i] for i in range(col_idx + 1, len(row_lineages)) if pd.notna(row_lineages[i])), None)
                
                lineage_range = []
                if prev_non_empty and next_non_empty:
                    lineage_range = [".".join(next_non_empty.split('.')[:i]) for i in range(len(next_non_empty.split('.')), 0, -1) 
                                     if ".".join(next_non_empty.split('.')[:i]).startswith(prev_non_empty)]
                elif prev_non_empty and not next_non_empty:
                    lineage_range = [prev_non_empty] + [l for l in lineages if l.startswith(prev_non_empty + '.')]
                elif not prev_non_empty and next_non_empty:
                    lineage_range = [".".join(next_non_empty.split('.')[:i]) for i in range(1, len(next_non_empty.split('.')) + 1)]
                
                if lineage_range:
                    metadata_filtered = metadata_human_select[
                        (metadata_human_select["pangolin_lineage"].isin(lineage_range)) & 
                        (metadata_human_select["season"] == season)
                    ]
                    if not metadata_filtered.empty:
                        sampled_row = metadata_filtered.sample(1)
                        genbank_accessions_season_filled.at[row_index, season] = sampled_row["genbank_accession"].values[0]
                        genbank_lineage_season_filled.at[row_index, season] = sampled_row["pangolin_lineage"].values[0]

    return genbank_accessions, genbank_date, genbank_accessions_season_filled, genbank_lineage_season_filled

def generate_training_data(fasta_path="./data/Processed/all_seq_msa_S_aa.fasta", 
                           notes_path='./data/supplyment/lineage_notes.txt', 
                           metadata_path='./data/public-latest.metadata.tsv', 
                           N=500, 
                           pathlen=6):

    sample_msa_S_aa = list(SeqIO.parse(fasta_path, "fasta"))
    for record in sample_msa_S_aa:
        record.id = record.id.split('_')[0]
        if record.seq.endswith('*'):
            record.seq = record.seq[:-1]
    lineage_notes_filtered = pd.read_csv(notes_path, sep='\t')
    metadata = pd.read_csv(metadata_path, sep='\t')
    metadata = metadata.drop(columns=['completeness']).iloc[:, :-2]

    metadata_human = metadata[metadata['host'] == 'Homo sapiens'].copy()
    metadata_human = metadata_human.drop(columns=['host'])
    metadata_human = metadata_human[~metadata_human["pangolin_lineage"].isna()]

    lineage_map = dict(zip(lineage_notes_filtered.iloc[:, 0], lineage_notes_filtered.iloc[:, 1]))
    metadata_human['pangolin_lineage'] = metadata_human['pangolin_lineage'].map(lineage_map)
    metadata_human = metadata_human[~metadata_human["pangolin_lineage"].isna()]

    accession_to_date = dict(zip(metadata_human['genbank_accession'], metadata_human['date']))
    accession_to_lineage = dict(zip(metadata_human['genbank_accession'], metadata_human['pangolin_lineage']))

    for record in sample_msa_S_aa:
        accession = record.id
        record_date = accession_to_date.get(accession, "Unknown") 
        record_lineage = accession_to_lineage.get(accession, "Unknown")  
        record.description = f"{record_date}|{record_lineage}"  

    genbank_accessions, _, _, _ = get_genbank_datasets(N=N, pathlen=pathlen)

    seq_dict  = {record.id: str(record.seq) for record in sample_msa_S_aa}
    date_dict = {record.id: record.description for record in sample_msa_S_aa}

    sample_dict = {}
    sample_date_dict = {}
    
    for idx, row in genbank_accessions.iterrows():
        ids = row[1:7] 
        sequences, dates, missing_ids = [], [], []

        for seq_id in ids:
            if seq_id in seq_dict:
                sequences.append(seq_dict[seq_id]) 
                dates.append(date_dict[seq_id])
            else:
                missing_ids.append(seq_id) 
        
        if not missing_ids:
            sample_dict[str(idx)] = sequences
            sample_date_dict[str(idx)] = dates

    sample_dict_samelen = {}
    sample_date_dict_samelen = {}
    for key, sequences in sample_dict.items():
        lengths = [len(seq) for seq in sequences]
        if len(set(lengths)) == 1:  
            sample_dict_samelen[key] = sequences
            sample_date_dict_samelen[key] = sample_date_dict[key]

    sample_dict = sample_dict_samelen
    sample_date_dict = sample_date_dict_samelen

    diff_positions_dict = {}
    for key, sequences in sample_dict.items():
        seq_len = len(sequences[0]) 
        diff_positions = []
        for i in range(seq_len):
            amino_acids_at_position = [seq[i] for seq in sequences]

            if len(set(amino_acids_at_position)) > 1 and "-" not in amino_acids_at_position and "X" not in amino_acids_at_position:
                diff_positions.append(i + 1)
        diff_positions_dict[key] = diff_positions

    sample_position_rows = []
    sample_date_rows = []

    for key, positions in diff_positions_dict.items():
        for pos in positions:
            start_pos = max(0, pos - 3)  
            end_pos = pos + 2  
            
            sequences = sample_dict[key]
            window_seqs = [seq[start_pos:end_pos] for seq in sequences]
            sample_position_rows.append([f"{key}|{pos}"] + window_seqs)
            
            dates = sample_date_dict[key]
            sample_date_rows.append([f"{key}|{pos}"] + dates)

    cols = ['Position'] + [f'Seq_{i+1}' for i in range(6)]
    sample_position = pd.DataFrame(sample_position_rows, columns=cols)
    sample_date = pd.DataFrame(sample_date_rows, columns=cols)

    def fill_date_format(date_str):
        if not isinstance(date_str, str): return np.nan
        if len(date_str) == 7: return date_str + "-01"
        if len(date_str) == 4: return np.nan
        return date_str

    for i in range(1, 7):
        sample_date[f'Seq_{i}'] = sample_date[f'Seq_{i}'].apply(fill_date_format)

    # 二分类
    sample_position_label = sample_position.copy()
    sample_position_label['Seq_6'] = sample_position_label.apply(
        lambda row: 0 if row['Seq_6'][2] == row['Seq_5'][2] else 1, axis=1
    )

    # 多分类
    sample_position_letter = sample_position.copy()
    sample_position_letter['Seq_6'] = sample_position_letter.apply(
        lambda row: row['Seq_6'][2], axis=1
    )
    return sample_position,sample_date,sample_position_label, sample_position_letter,sample_dict

def process_and_save_3gram_datasets(sample_position_label, sample_position_letter, sample_date, 
                                    protvec_path):
    
    protVec_100d_3grams = pd.read_csv(protvec_path, sep='\t')
    triplet_dict = {words: index for index, words in enumerate(protVec_100d_3grams['words'])}

    def sequence_to_triplet_nums(seq, triplet_dict):
        triplets_in_seq = [seq[i:i+3] for i in range(3)]
        return [triplet_dict.get(triplet, -1) for triplet in triplets_in_seq] 

    sample_position_label_num = sample_position_label.copy()
    seq_columns = ['Seq_1', 'Seq_2', 'Seq_3', 'Seq_4', 'Seq_5']
    for col in seq_columns:
        sample_position_label_num[col] = sample_position_label_num[col].apply(lambda seq: sequence_to_triplet_nums(seq, triplet_dict))

    for col in ["Seq_1","Seq_2","Seq_3","Seq_4","Seq_5"]:
        sample_position_label_num[col] = sample_position_label_num[col].apply(lambda x: [9047 if i == -1 else i for i in x])
        sample_position_label_num.loc[:, col] = sample_position_label_num[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    sample_position_letter_num = sample_position_letter.copy()
    seq_columns = ['Seq_1', 'Seq_2', 'Seq_3', 'Seq_4', 'Seq_5']
    for col in seq_columns:
        sample_position_letter_num[col] = sample_position_letter_num[col].apply(lambda seq: sequence_to_triplet_nums(seq, triplet_dict))

    for col in ["Seq_1","Seq_2","Seq_3","Seq_4","Seq_5"]:
        sample_position_letter_num[col] = sample_position_letter_num[col].apply(lambda x: [9047 if i == -1 else i for i in x])

    sample_position_label_num_dedup = sample_position_label_num.drop_duplicates(subset=sample_position_label_num.columns[1:])

    for col in sample_position_label_num_dedup.columns[1:]:
        sample_position_label_num_dedup.loc[:, col] = sample_position_label_num_dedup[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)
    
    sample_date_dedup = sample_date[sample_date['Position'].isin(sample_position_label_num_dedup['Position'])]

    sample_position_label_num_dedup.columns = ['Position'] + [0,1,2,3,4,"y"]
    sampled_label_dedup = sample_position_label_num_dedup.reindex(columns=['Position', "y",0, 1, 2, 3, 4])

    sample_date_dedup.columns = ['Position'] + [0,1,2,3,4,"y"]
    sampled_df_date_dedup = sample_date_dedup.reindex(columns=['Position', "y",0, 1, 2, 3, 4])

    sampled_label_date_dedup_merge = pd.concat([sampled_label_dedup, sampled_df_date_dedup['y'].str.split("|").str[0]], axis=1)
    sampled_label_date_dedup_merge.columns = ['Position', "y",0, 1, 2, 3, 4, 'date']
    sampled_label_date_dedup_merge['date_month'] = sampled_label_date_dedup_merge['date'].str[:7]
    sampled_label_date_dedup_merge = sampled_label_date_dedup_merge.sort_values(by="date").reset_index(drop=True) 

    test_df_label = sampled_label_date_dedup_merge[sampled_label_date_dedup_merge['date'] >= '2022-03-25']
    train_df_label = sampled_label_date_dedup_merge[sampled_label_date_dedup_merge['date'] < '2022-03-25']

    part1len = len(test_df_label[test_df_label['y'] == 1])
    part1 = test_df_label.loc[test_df_label['y'] == 1].sample(n=min(500,part1len))
    part0 = test_df_label.loc[test_df_label['y'] == 0].sample(n=1000-len(part1))
    test_df_label_1k = pd.concat([part1, part0]).sample(frac=1)

    part1len = len(train_df_label[train_df_label['y'] == 1])
    part1 = train_df_label.loc[train_df_label['y'] == 1].sample(n=min(2000,part1len))
    part0 = train_df_label.loc[train_df_label['y'] == 0].sample(n=4000-len(part1))
    train_df_label_4k = pd.concat([part1, part0]).sample(frac=1)

    train_df_letter_4k = train_df_label_4k.copy()
    train_df_letter_4k['y'] = train_df_letter_4k['Position'].map(sample_position_letter_num.set_index('Position')['Seq_6'])

    test_df_letter_1k = test_df_label_1k.copy()
    test_df_letter_1k['y'] = test_df_letter_1k['Position'].map(sample_position_letter_num.set_index('Position')['Seq_6'])

    save_path = './data/Processed/Covid19/3gram/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # without position
    train_df_letter_4k.iloc[:,1:-2].to_csv(save_path + 'letter_4k_train.csv', index=False)
    test_df_letter_1k.iloc[:,1:-2].to_csv(save_path + 'letter_1k_test.csv', index=False)
    train_df_label_4k.iloc[:,1:-2].to_csv(save_path + 'label_4k_train.csv', index=False)
    test_df_label_1k.iloc[:,1:-2].to_csv(save_path + 'label_1k_test.csv', index=False)
    # with position
    train_df_letter_4k.to_csv(save_path + 'index_letter_4k_train.csv', index=False)
    test_df_letter_1k.to_csv(save_path + 'index_letter_1k_test.csv', index=False)
    train_df_label_4k.to_csv(save_path + 'index_label_4k_train.csv', index=False)
    test_df_label_1k.to_csv(save_path + 'index_label_1k_test.csv', index=False)

    return train_df_letter_4k, test_df_letter_1k, train_df_label_4k, test_df_label_1k

def save_esm_assets(train_df_label_4k, test_df_label_1k, sample_dict):
    sequence_dict_train = {}
    sequence_dict_test = {}

    for idx, row in train_df_label_4k.iterrows():
        position = row['Position']
        key = position.split('|')[0]
        for col in [0, 1, 2, 3, 4]:
            if key in sample_dict:
                sequence = sample_dict[key][col]
                new_key = f"{position}.{col}"
                sequence_dict_train[new_key] = sequence

    for idx, row in test_df_label_1k.iterrows():
        position = row['Position']
        key = position.split('|')[0]
        for col in [0, 1, 2, 3, 4]:
            if key in sample_dict:
                sequence = sample_dict[key][col]
                new_key = f"{position}.{col}"
                sequence_dict_test[new_key] = sequence

    unique_sequence_dict_train = {}
    unique_sequence_dict_test = {}
    seen_sequences_train = set()
    seen_sequences_test = set()

    for key, value in sequence_dict_train.items():
        if value not in seen_sequences_train:
            unique_sequence_dict_train[key] = value
            seen_sequences_train.add(value)

    for key, value in sequence_dict_test.items():
        if value not in seen_sequences_test:
            unique_sequence_dict_test[key] = value
            seen_sequences_test.add(value)

    value_to_keys_train = defaultdict(list)
    value_to_keys_test = defaultdict(list)

    for key, value in sequence_dict_train.items():
        value_to_keys_train[value].append(key)
    for key, value in sequence_dict_test.items():
        value_to_keys_test[value].append(key)

    duplicate_values_train = {value: keys for value, keys in value_to_keys_train.items() if len(keys) >= 1}
    duplicate_values_test = {value: keys for value, keys in value_to_keys_test.items() if len(keys) >= 1}

    esm_path = "./data/Processed/Covid19/ESM/"
    if not os.path.exists(esm_path):
        os.makedirs(esm_path)

    output_fasta_train = esm_path + "esm_input_train.fasta"
    with open(output_fasta_train, 'w') as fasta_file:
        for key, sequence in unique_sequence_dict_train.items():
            fasta_file.write(f">{key}\n")
            fasta_file.write(f"{sequence}\n")

    output_fasta_test = esm_path + "esm_input_test.fasta"
    with open(output_fasta_test, 'w') as fasta_file:
        for key, sequence in unique_sequence_dict_test.items():
            fasta_file.write(f">{key}\n")
            fasta_file.write(f"{sequence}\n")

    key2keys_test = {}
    for key, value in unique_sequence_dict_test.items():
        if value in duplicate_values_test:
            key2keys_test[key] = duplicate_values_test[value]

    key2keys_train = {}
    for key, value in unique_sequence_dict_train.items():
        if value in duplicate_values_train:
            key2keys_train[key] = duplicate_values_train[value]

    json_path_test = esm_path + 'key2keys_branch1225_test.json'
    with open(json_path_test, 'w') as json_file:
        json.dump(key2keys_test, json_file, indent=4)

    json_path_train = esm_path + 'key2keys_branch1225_train.json'
    with open(json_path_train, 'w') as json_file:
        json.dump(key2keys_train, json_file, indent=4)

if __name__ == "__main__":
    genbank_accessions, genbank_date, genbank_accessions_season_filled, genbank_lineage_season_filled = get_genbank_datasets()
    sample_position,sample_date,sample_position_label, sample_position_letter,sample_dict = generate_training_data()
    train_df_letter_4k, test_df_letter_1k, train_df_label_4k, test_df_label_1k = process_and_save_3gram_datasets(sample_position_label, sample_position_letter, sample_date,protvec_path='./data/supplyment/protVec_100d_3grams.csv')
    save_esm_assets(train_df_label_4k, test_df_label_1k, sample_dict)