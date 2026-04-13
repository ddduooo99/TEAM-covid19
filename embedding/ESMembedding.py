import torch
from esm import pretrained
from tqdm import tqdm
from Bio import SeqIO
import pickle


AAs = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',]

path = "./data/Processed/Covid19/ESM/esm_input_train.fasta"
esm_input = list(SeqIO.parse(path, "fasta"))
for record in esm_input:
    record.seq = record.seq.replace('*', '-')
for record in esm_input:
        record.seq = ''.join([aa if aa in AAs else 'X' for aa in record.seq])

def embedding_esm(esm_input,model,batch_converter):
    # 准备数据
    data = [("protein1", esm_input),]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # 获取 embedding 矩阵
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33])  # 表示我们想要第33层的输出嵌入（你可以选择其他层）
    token_embeddings = results['representations'][33]  # 获取第33层的嵌入矩阵
    return token_embeddings[0,1:-1,:]


model_locations = [
# 'esm1b_t33_650M_UR50S',
'esm1v_t33_650M_UR90S_1',
# 'esm1v_t33_650M_UR90S_2',
# 'esm1v_t33_650M_UR90S_3',
# 'esm1v_t33_650M_UR90S_4',
# 'esm1v_t33_650M_UR90S_5',
]

for model_location in model_locations:
    # 加载模型和字母表
    model, alphabet = pretrained.load_model_and_alphabet(model_location)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # 获取每个模型的 batch converter
    batch_converter = alphabet.get_batch_converter()

    # 用于存储所有序列的 embedding
    embeddings_dict = {}

    for record in tqdm(esm_input, desc=f"Processing {model_location}"):
        sequence = str(record.seq)
        sequence_id = record.id
        # print(f"Processing {record.id}")
        # if len(sequence) != 1273:
        #     print(f"Skipping {sequence_id} due to length {len(sequence)}")
        #     continue
        segment1 = sequence[:1000]
        segment2 = sequence[-1000:]

        # 分别对两个部分进行 embedding
        token_embedding_1 = embedding_esm(segment1, model, batch_converter)
        token_embedding_2 = embedding_esm(segment2, model, batch_converter)

        first_part = token_embedding_1[:636, :]
        second_part = token_embedding_2[-637:, :]
        token_embeddings = torch.cat((first_part, second_part), dim=0)

        # 存储 embedding 到字典中，以 sequence_id 为键
        embeddings_dict[sequence_id] = token_embeddings.cpu().numpy()
    
    # 保存 embedding 到文件
    with open(f"./embedding/embedding_train_{model_location}.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)