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
        sequence_feature = load_features(sequence_name, sequence, self.mean, self.std, self.blosum)  
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
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
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

    def forward(self, x, adj):              # x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)                # x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))    # output.shape = (seq_len, GCN_OUTPUT_DIM)
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

    def forward(self, input):               # input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))     # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)                     # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)       # attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):                                              # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)                                                # x.shape = (seq_len, GAT_OUTPUT_DIM)

        x = x.unsqueeze(0).float()                                          # x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)                                             # att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x                                    # output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))   # output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


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


def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name,map_location='cpu'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", np.sqrt(epoch_loss_test_avg))
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test auc: ", result_test['auc'])
        print("Test mcc: ", result_test['mcc'])
        print("Test sensitivity: ", result_test['sensitivity'])
        print("Test specificity: ", result_test['specificity'])

        test_result[model_name] = [
            np.sqrt(epoch_loss_test_avg),
            result_test['pearson'],
            result_test['r2'],
            result_test['binary_acc'],
            result_test['precision'],
            result_test['recall'],
            result_test['f1'],
            result_test['auc'],
            result_test['mcc'],
            result_test['sensitivity'],
            result_test['specificity'],
        ]

        test_detail_dataframe = pd.DataFrame({'gene': test_name, 'solubility': test_true, 'prediction': test_pred})
        test_detail_dataframe.sort_values(by=['gene'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'pearson', 'r2', 'binary_acc', 'precision',
                                                            'recall', 'f1', 'auc', 'mcc', 'sensitivity', 'specificity'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')


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


if __name__ == "__main__":
    test_dataframe = pd.read_csv(Dataset_Path + 'eSol_test.csv', sep=',')
    test(test_dataframe) 
   