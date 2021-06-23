import numpy as np
import pandas as pd


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_blosum(wkdir='.'):
    with open(f'{wkdir}/Data/common/BLOSUM62_dim23.txt', 'r') as f:
        result = dict()
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result


def load_aaphy7(wkdir='.'):
    with open(f'{wkdir}/Data/common/aa_phy7', 'r') as f:
        result = dict()
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [float(i) for i in line[1:]]
    return result


def load_mean_std(wkdir='.'):
    # (23,)
    blosum_mean = np.load(f'{wkdir}/Data/common/eSol_blosum_mean.npy')
    blosum_std = np.load(f'{wkdir}/Data/common/eSol_blosum_std.npy')

    # (71,)
    oneD_mean = np.load(f'{wkdir}/Data/common/eSol_oneD_mean.npy')
    oneD_std = np.load(f'{wkdir}/Data/common/eSol_oneD_std.npy')

    mean = np.concatenate([blosum_mean, oneD_mean])
    std = np.concatenate([blosum_std, oneD_std])

    return mean, std


def load_pssm(wkdir, fname, seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(f'{wkdir}/Data/source/{fname}.pssm', 'r') as f:
        tmp_pssm = pd.read_csv(f, delim_whitespace=True, names=pssm_col_names).dropna().values[:, 2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm


def load_hhm(wkdir, fname, seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(f'{wkdir}/Data/source/{fname}.hhm', 'r') as f:
        hhm = pd.read_csv(f, delim_whitespace=True, names=hhm_col_names)
    pos1 = (hhm['0'] == 'HMM').idxmax() + 3
    hhm = hhm[pos1:-1].values[:, :num_hhm_cols].reshape([-1, 44])
    hhm[hhm == '*'] = '9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:, 2:-12].astype(float)


def spd3_feature_sincos(x, seq):
    ASA = x[:, 0]
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_div = np.array([dict_rnam1_ASA[i] for i in seq])
    ASA = (ASA / ASA_div)[:, None]
    angles = x[:, 1:5]
    HSEa = x[:, 5:7]
    HCEprob = x[:, -3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles), np.cos(angles)], 1)
    return np.concatenate([ASA, angles, HSEa, HCEprob], 1)


def load_spd33(wkdir, fname, seq):
    with open(f'{wkdir}/Data/source/{fname}.spd33', 'r') as f:
        spd3_features = pd.read_csv(f, delim_whitespace=True).values[:, 3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features, seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('SPD3 file is in wrong format or incorrect!')
    return tmp_spd3


def load_spotcon(wkdir, fname, sequence):
    with open(f'{wkdir}/Data/source/{fname}.spotcon', 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index < 5:
            continue
        elif index == 5:
            matrix = np.ones((len(sequence), len(sequence)), dtype=float)
            mask1 = np.tril(np.ones((len(sequence), len(sequence)), dtype=float), -3)
            mask2 = np.triu(np.ones((len(sequence), len(sequence)), dtype=float), 3)
            mask = mask1 + mask2
            matrix -= mask
        else:
            data = line.strip().split()
            matrix[int(data[0])][int(data[1])] = float(data[2])
            matrix[int(data[1])][int(data[0])] = float(data[2])
    return matrix


def load_node_features(wkdir, sequence_name, mean, std):
    matrix = np.load(f'{wkdir}/Data/generate/{sequence_name}_oneD.npy')
    matrix = (matrix - mean) / std
    return matrix


def load_edge_features(wkdir, sequence_name):
    matrix = np.load(f'{wkdir}/Data/generate/{sequence_name}_twoD.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


if __name__ == '__main__':
    pass
