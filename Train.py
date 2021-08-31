import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

# path
Dataset_Path = './Data/'
Model_Path = './Model/'
Result_Path = './Result/'

amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 25
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 1
NUM_CLASSES = 1

# GCN parameters
GCN_FEATURE_DIM = 91
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})


def load_features(sequence_name, sequence, mean, std, blosum):
    # len(sequence) * 23
    blosum_matrix = np.array([blosum[i] for i in sequence])
    # len(sequence) * 71
    oneD_matrix = np.load(Dataset_Path + 'node_features/' + sequence_name + '.npy')
    # len(sequence) * 94
    feature_matrix = np.concatenate([blosum_matrix, oneD_matrix], axis=1)
    feature_matrix = (feature_matrix - mean) / std
    part1 = feature_matrix[:,0:20]
    part2 = feature_matrix[:,23:]
    # len(sequence) * 91
    feature_matrix = np.concatenate([part1,part2],axis=1)
    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'edge_features/' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


def load_values():
    # (23,)
    blosum_mean = np.load(Dataset_Path + 'eSol_blosum_mean.npy')
    blosum_std = np.load(Dataset_Path + 'eSol_blosum_std.npy')

    # (71,)
    oneD_mean = np.load(Dataset_Path + 'eSol_oneD_mean.npy')
    oneD_std = np.load(Dataset_Path + 'eSol_oneD_std.npy')

    mean = np.concatenate([blosum_mean, oneD_mean])
    std = np.concatenate([blosum_std, oneD_std])

    return mean, std


def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result


class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['gene'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['solubility'].values
        self.mean, self.std = load_values()
        self.blosum = load_blosum()

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        # L * 91
        sequence_feature = load_features(sequence_name, sequence, self.mean, self.std, self.blosum)
        # L * L
        sequence_graph = load_graph(sequence_name)
        return sequence_name, sequence, label, sequence_feature, sequence_graph

    def __len__(self):
        return len(self.labels)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):  											# x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)  												# x.shape = (seq_len, GAT_OUTPUT_DIM)

        x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))  	# output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


def train_one_epoch(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        _, _, labels, sequence_features, sequence_graphs = data

        sequence_features = torch.squeeze(sequence_features)
        sequence_graphs = torch.squeeze(sequence_graphs)

        if torch.cuda.is_available():
            features = Variable(sequence_features.cuda())
            graphs = Variable(sequence_graphs.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_features)
            graphs = Variable(sequence_graphs)
            y_true = Variable(labels)

        y_pred = model(features, graphs)
        y_true = y_true.float()

        # calculate loss
        loss = model.criterion(y_pred, y_true)

        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, sequence_features, sequence_graphs = data

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)
                y_true = Variable(labels)

            y_pred = model(features, graphs)
            y_true = y_true.float()

            loss = model.criterion(y_pred, y_true)
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, valid_true, valid_pred, valid_name


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    train_losses = []
    train_pearson = []
    train_r2 = []
    train_binary_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_auc = []
    train_mcc = []
    train_sensitivity = []
    train_specificity = []

    valid_losses = []
    valid_pearson = []
    valid_r2 = []
    valid_binary_acc = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []
    valid_auc = []
    valid_mcc = []
    valid_sensitivity = []
    valid_specificity = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", np.sqrt(epoch_loss_train_avg))
        print("Train pearson:", result_train['pearson'])
        print("Train r2:", result_train['r2'])
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train precision: ", result_train['precision'])
        print("Train recall: ", result_train['recall'])
        print("Train F1: ", result_train['f1'])
        print("Train auc: ", result_train['auc'])
        print("Train mcc: ", result_train['mcc'])
        print("Train sensitivity: ", result_train['sensitivity'])
        print("Train specificity: ", result_train['specificity'])

        train_losses.append(np.sqrt(epoch_loss_train_avg))
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        train_binary_acc.append(result_train['binary_acc'])
        train_precision.append(result_train['precision'])
        train_recall.append(result_train['recall'])
        train_f1.append(result_train['f1'])
        train_auc.append(result_train['auc'])
        train_mcc.append(result_train['mcc'])
        train_sensitivity.append(result_train['sensitivity'])
        train_specificity.append(result_train['specificity'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid auc: ", result_valid['auc'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Valid sensitivity: ", result_valid['sensitivity'])
        print("Valid specificity: ", result_valid['specificity'])

        valid_losses.append(np.sqrt(epoch_loss_valid_avg))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])
        valid_binary_acc.append(result_valid['binary_acc'])
        valid_precision.append(result_valid['precision'])
        valid_recall.append(result_valid['recall'])
        valid_f1.append(result_valid['f1'])
        valid_auc.append(result_valid['auc'])
        valid_mcc.append(result_valid['mcc'])
        valid_sensitivity.append(result_valid['sensitivity'])
        valid_specificity.append(result_valid['specificity'])

        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_detail_dataframe = pd.DataFrame({'gene': valid_name, 'solubility': valid_true, 'prediction': valid_pred})
            valid_detail_dataframe.sort_values(by=['gene'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')

    # save calculation information
    result_all = {
        'Train_loss': train_losses,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        'Train_binary_acc': train_binary_acc,
        'Train_precision': train_precision,
        'Train_recall': train_recall,
        'Train_f1': train_f1,
        'Train_auc': train_auc,
        'Train_mcc': train_mcc,
        'Train_sensitivity': train_sensitivity,
        'Train_specificity': train_specificity,
        'Valid_loss': valid_losses,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        'Valid_binary_acc': valid_binary_acc,
        'Valid_precision': valid_precision,
        'Valid_recall': valid_recall,
        'Valid_f1': valid_f1,
        'Valid_auc': valid_auc,
        'Valid_mcc': valid_mcc,
        'Valid_sensitivity': valid_sensitivity,
        'Valid_specificity': valid_specificity,
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')


def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    pearson = pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auc = metrics.roc_auc_score(binary_true, y_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'pearson': pearson,
        'r2': r2,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result


def cross_validation(all_dataframe,fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['gene'].values
    sequence_labels = all_dataframe['solubility'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model()
        if torch.cuda.is_available():
            model.cuda()

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1


if __name__ == "__main__":
    train_dataframe = pd.read_csv(Dataset_Path + 'eSol_train.csv',sep=',')
    cross_validation(train_dataframe,fold_number=5)
