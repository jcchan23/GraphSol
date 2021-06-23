import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 25
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 1
LEAKY_SLOPE = 0.2

# GCN parameters
GCN_FEATURE_DIM = 94
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4


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
        support = input @ self.weight   # X * W
        output = adj @ support          # A * X * W
        if self.bias is not None:       # A * X * W + b
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
        self.relu1 = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        self.relu2 = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)

    def forward(self, x, adj):            # x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)              # x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))  # output.shape = (seq_len, GCN_OUTPUT_DIM)
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

    def forward(self, input):             # input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))   # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)                   # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)     # attention.shape = (1, attention_hops, seq_len)
        return attention


class GraphSol(nn.Module):
    def __init__(self):
        super(GraphSol, self).__init__()

        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(GCN_OUTPUT_DIM, 1)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):
        # x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)
        # x.shape = (seq_len, GCN_OUTPUT_DIM) ==> (1, seq_len, GCN_OUTPUT_DIM)
        x = x.unsqueeze(0).float()
        # att.shape = (1, ATTENTION_HEADS, seq_len)
        att = self.attention(x)
        # node_feature_embedding.shape = (1, ATTENTION_HEADS, GCN_OUTPUT_DIM)
        node_feature_embedding = att @ x
        # node_feature_embedding_avg.shape = (1, GCN_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding, 1) / self.attention.n_heads
        # output.shape = (1, 1)
        output = torch.sigmoid(self.fc_final(node_feature_embedding_avg))
        return output.squeeze(0)
