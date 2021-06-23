from torch.utils.data import Dataset
from utils import load_mean_std, load_node_features, load_edge_features


class ProDataset(Dataset):

    def __init__(self, wkdir, sequence_dict):
        self.wkdir = wkdir
        self.names = list(sequence_dict.keys())
        self.sequences = list(sequence_dict.values())
        self.mean, self.std = load_mean_std(self.wkdir)

    def __getitem__(self, index):
        name = self.names[index]
        sequence = self.sequences[index]
        # L * 91
        node_features = load_node_features(self.wkdir, name, self.mean, self.std)
        # L * L
        edge_features = load_edge_features(self.wkdir, name)
        return name, sequence, node_features, edge_features

    def __len__(self):
        return len(self.names)

