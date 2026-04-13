import os, sys

# sys.path.append(os.path.abspath("/home/zh/codes/rnn_virus_source_code"))
import models
import train_model, train_model_multi
import make_dataset
import build_features
import utils
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_train_test_data(isletter=True):
    if subtype_flag == 3:
        if isletter:
            data_path = './data/Processed/Covid19/3gram/'
            data_set = './data/Processed/Covid19/3gram/letter_4k1k'
        else:
            data_path = './data/Processed/Covid19/3gram/'
            data_set = './data/Processed/Covid19/3gram/label_4k1k'
    elif subtype_flag == 4:
        if isletter:
            data_path = './data/Processed/Covid19/ESM/'
            data_set = './data/Processed/Covid19/ESM/esm1v1_letter'

        else:
            data_path = './data/Processed/Covid19/ESM/'
            data_set = './data/Processed/Covid19/ESM/esm1v1_label'


    train_trigram_vecs, train_labels = utils.read_data_esm_cat(data_set + '_train.csv')
    test_trigram_vecs, test_labels = utils.read_data_esm_cat(data_set + '_test.csv')


    if isletter:
        # 将 train 和 test 标签合并在一起进行编码
        all_labels =np.concatenate([train_labels, test_labels])
        label_encoder= LabelEncoder()
        label_encoder.fit(all_labels)
        # 对 train 和 test 标签进行转换
        train_labels= label_encoder.transform(train_labels)
        test_labels =label_encoder.transform(test_labels)

    X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
    Y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)

    # give weights for imbalanced dataset
    _, counts = np.unique(Y_train, return_counts=True)
    train_counts = max(counts)
    train_imbalance = max(counts) / Y_train.shape[0]
    _, counts = np.unique(Y_test, return_counts=True)
    test_counts = max(counts)
    test_imbalance = max(counts) / Y_test.shape[0]

    print('Class imbalances:')
    print(' Training %.3f' % train_imbalance)
    print(' Testing  %.3f' % test_imbalance)
    return X_train, Y_train, X_test, Y_test

def main(X_train, Y_train, X_test, Y_test, isletter=True):
    parameters = {

        'model': model,
        'hidden_size': 512,
        'dropout_p': 0.0001,
        'dropout_w': 0.2,
        'learning_rate': 1e-2,
        'batch_size': 256,
        'num_of_epochs': 200
    }
    if parameters['model'] == 'svm':
        window_size = 1
        train_model.svm_baseline(
            build_features.reshape_to_linear(X_train, window_size=window_size), Y_train,
            build_features.reshape_to_linear(X_test, window_size=window_size), Y_test)
    elif parameters['model'] == 'random forest':
        window_size = 1
        train_model.random_forest_baseline(
            build_features.reshape_to_linear(X_train, window_size=window_size), Y_train,
            build_features.reshape_to_linear(X_test, window_size=window_size), Y_test)
    elif parameters['model'] == 'logistic regression':
        window_size = 1
        train_model.logistic_regression_baseline(
            build_features.reshape_to_linear(X_train, window_size=window_size), Y_train,
            build_features.reshape_to_linear(X_test, window_size=window_size), Y_test)
    else:
        input_dim = X_train.shape[2]
        seq_length = X_train.shape[0]
        output_dim =  len(torch.unique(torch.cat([Y_train, Y_test])))
        if parameters['model'] == 'lstm':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='LSTM')
        elif parameters['model'] == 'gru':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='GRU')
        elif parameters['model'] == 'rnn':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='RNN')
        elif parameters['model'] == 'attention':
            net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'],
                                        parameters['dropout_p'])
        elif parameters['model'] == 'da-rnn':
            net = models.DaRnnModel(seq_length, input_dim, output_dim, parameters['hidden_size'],
                                    parameters['dropout_p'])
        elif parameters['model'] == 'transformer':
            net = models.TransformerModel(input_dim, output_dim, parameters['dropout_p'],nhead=2)
        elif parameters['model'] == 'postrans':
            net = models.PosTrans(input_dim, output_dim, parameters['dropout_w'], head_num=2, n_layer=4)
        
        # use gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        # use gpu
        if not isletter:
            bst = train_model.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'],
                              parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'])
        else:
            bst = train_model_multi.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'],
                              parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'])
        return bst


if __name__ == '__main__':
    subtype = ['COV19', 'COV19ESM']
    subtype_flag = make_dataset.subtype_selection(subtype[3])

    models_list = ['postrans', 'rnn', 'lstm', 'gru', 'attention', 'transformer', 'da-rnn']
    res = {}
    results = []  # 保存结果
    X_train, Y_train, X_test, Y_test = get_train_test_data(isletter=False)

    for i in range(4):  # 跑10次
        seed = i + 1  # 每次用不同的种子
        print(f'\nRunning with seed: {seed}')


        for model_name in models_list:
            try:
                print("\n")
                print(f"Experimental results with model {model_name} on subtype_flag {subtype_flag} using seed {seed}:")
                bst_result = main(X_train, Y_train, X_test, Y_test, model_name,isletter=False)
                res[model_name] = bst_result

                results.append({
                    "seed": seed,
                    "model": model_name,
                    "result": bst_result
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
        for name, bst in res.items():
            print(name, bst)

    # 打印并保存所有结果
    for result in results:
        print(f"Seed: {result['seed']}, Model: {result['model']}, Result: {result['result']}")











