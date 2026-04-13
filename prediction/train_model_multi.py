import torch
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression

import pandas as pd


# from scipy import interp

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc, fscore, val_fscore, cell_type):
    """
    Plots the loss and accuracy for training and validation over epochs.
    Also plots the logits for a small batch over epochs.
    """
    plt.style.use('ggplot')

    # Plot losses
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 3)
    plt.plot(fscore, 'b', label='Training')
    plt.plot(val_fscore, 'r', label='Validation')
    plt.title('F-Score')
    plt.legend()

    # plt.show()
    plt.savefig(f'./data/figure/loss_fig_{cell_type}.png', dpi=350)
    print(f'plot save to ./data/figure/loss_fig_{cell_type}.png')


def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def calculate_prob(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    pred_probe, _ = prob.topk(1)
    return pred_probe


def verify_model(model, X, Y, batch_size):
    """
    Checks the loss at initialization of the model and asserts that the
    training examples in a batch aren't mixed together by backpropagating.
    """
    print('Sanity checks:')
    criterion = torch.nn.CrossEntropyLoss()
    scores, _ = model(X, model.init_hidden(Y.shape[0]))
    print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))

    mini_batch_X = X[:, :batch_size, :]
    mini_batch_X.requires_grad_()
    # 修改之前
    # criterion = torch.nn.MSELoss()

    # 修改之后
    criterion = torch.nn.CrossEntropyLoss()  # 对分类任务使用交叉熵损失

    scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

    non_zero_idx = 1
    perfect_scores = [[0, 0] for i in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(mini_batch_X.shape[0]):
        for j in range(mini_batch_X.shape[1]):
            if sum(mini_batch_X.grad[i, j] != zero_tensor):
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

    mini_batch_X.detach()
    print(' Backpropagated dependencies OK')


def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention, cell_type):
    """
    Training loop for a model utilizing hidden states.

    verify enables sanity checks of the model.
    epochs decides the number of training iterations.
    learning rate decides how much the weights are updated each iteration.
    batch_size decides how many examples are in each mini batch.
    show_attention decides if attention weights are plotted.
    """
    print_interval = epochs // 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples / batch_size)

    if verify:
        verify_model(model, X, Y, batch_size)
    all_losses = []
    all_val_losses = []
    all_accs = []
    all_val_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_val_fscores = []
    all_mccs = []

    best_val_loss = 100000000.0
    best_val_acc = 0.0
    best_val_pre = 0.0
    best_val_rec = 0.0
    best_val_fscore = 0.0
    best_val_mcc = 0.0
    best_epoch_index = 0

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_train_predictions = []
        all_train_true = []

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            repackage_hidden(hidden)

            X_batch = X[:, count:count + batch_size, :]
            Y_batch = Y[count:count + batch_size]

            scores, _ = model(X_batch, hidden)

            loss = criterion(scores, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 修改之前
            # predictions = predictions_from_output(scores)
            # 修改之后
            predictions = torch.argmax(scores, dim=1)  # logits 转为类别索引
            
            all_train_predictions.extend(predictions.cpu().numpy())
            all_train_true.extend(Y_batch.cpu().numpy())

            running_loss += loss.item()

        # Calculate metrics on the entire training dataset for this epoch
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        epoch_acc = accuracy_score(all_train_true, all_train_predictions)
        all_accs.append(epoch_acc)

        epoch_pre = precision_score(all_train_true, all_train_predictions, average='micro')
        all_pres.append(epoch_pre)

        epoch_rec = recall_score(all_train_true, all_train_predictions, average='micro')
        all_recs.append(epoch_rec)

        epoch_fscore = f1_score(all_train_true, all_train_predictions, average='micro')
        all_fscores.append(epoch_fscore)

        epoch_mcc = matthews_corrcoef(all_train_true, all_train_predictions)
        all_mccs.append(epoch_mcc)

        # Validation metrics
        with torch.no_grad():
            model.eval()
            test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))

            predictions = predictions_from_output(test_scores)
            predictions = predictions.view_as(Y_test)

            y_true = Y_test.cpu().numpy()
            predictions = predictions.cpu().numpy()

            val_loss = criterion(test_scores, Y_test).item()
            val_acc = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='micro')
            recall = recall_score(y_true, predictions, average='micro')
            fscore = f1_score(y_true, predictions, average='micro')
            mcc = matthews_corrcoef(y_true, predictions)

            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)
            all_val_fscores.append(fscore)

            if val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_pre = precision
                best_val_rec = recall
                best_val_fscore = fscore
                best_val_mcc = mcc
                best_epoch_index = epoch

    # plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, all_fscores, all_val_fscores, cell_type)
    bst = 'Best results: %d \t V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
        best_epoch_index, best_val_loss, best_val_acc, best_val_pre, best_val_rec, best_val_fscore, best_val_mcc)
    print(bst)
    return bst


def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced', probability=True).fit(X, Y)

    # Training metrics
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    # Validation metrics
    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    pd.DataFrame(Y).to_csv("./output/Y_train.txt", sep="\t", index=False)
    pd.DataFrame(train_predictions).to_csv("./output/Y_train_pred.txt", sep="\t",
                                           index=False)
    pd.DataFrame(Y_pred).to_csv("./output/Y_test_pred.txt", sep="\t", index=False)
    pd.DataFrame(Y_test).to_csv("./output/Y_test.txt", sep="\t", index=False)

    # Print metrics
    print('SVM baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    # Save results if method is specified
    if method is not None:
        with open('./reports/results/{}_SVM.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)



def random_forest_baseline(X, Y, X_test, Y_test, method=None):
    clf = ensemble.RandomForestClassifier().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Random Forest baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    if method is not None:
        with open('./reports/results/{}_RF.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)



def knn_baseline(X, Y, X_test, Y_test, method=None):
    clf = KNeighborsClassifier().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('KNN baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    if method is not None:
        with open('./reports/results/{}_KNN.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)


def bayes_baseline(X, Y, X_test, Y_test, method=None):
    clf = GaussianNB().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Bayes baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

def logistic_regression_baseline(X, Y, X_test, Y_test, method=None):
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X), average='micro')
    train_rec = recall_score(Y, clf.predict(X), average='micro')
    train_fscore = f1_score(Y, clf.predict(X), average='micro')
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))