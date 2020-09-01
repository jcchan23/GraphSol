import os
import numpy as np
import pandas as pd
from tqdm import tqdm

with open('./Data/aa_phy7','r') as f:
    pccp = f.read().splitlines()
    pccp = [i.split() for i in pccp]
    pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}


def read_pccp(seq):
    return np.array([pccp_dic[i] for i in seq])


def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm


def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)


def spd3_feature_sincos(x,seq):
    ASA = x[:,0]
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                        185, 160, 145, 180, 225, 115, 140, 155, 255, 230,1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
    ASA = (ASA/ASA_div)[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,-3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)


def read_spd33(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('SPD3 file is in wrong format or incorrect!')
    return tmp_spd3


def read_fasta(fname):
    with open(fname,'r') as f:
        sequence = f.read().splitlines()[1]
    return sequence


def get_matrix():
    for gene in tqdm(os.listdir('./Data/fasta/')):
        sequence = read_fasta('./Data/fasta/' + gene)
        pssm = read_pssm('./Data/pssm/' + gene + '.pssm',sequence)
        hhm = read_hhm('./Data/hhm/' + gene + '.hhm',sequence)
        spd33 = read_spd33('./Data/spd33/' + gene + '.spd33',sequence)
        aaph7 = read_pccp(sequence)
        matrix = np.concatenate([pssm,hhm,spd33,aaph7],axis=1)
        np.save('./Data/node_features/'+gene+'.npy',matrix)


def cal_mean_std():
    total_length = 0
    mean = np.zeros(71)
    mean_square = np.zeros(71)
    for name in tqdm(os.listdir('./Data/node_features/')):
        matrix = np.load('./Data/node_features/' + name)
        total_length += matrix.shape[0]
        mean += np.sum(matrix, axis=0)
        mean_square += np.sum(np.square(matrix), axis=0)

    mean /= total_length  # EX
    mean_square /= total_length  # E(X^2)
    std = np.sqrt(np.subtract(mean_square, np.square(mean)))  # DX = E(X^2)-(EX)^2, std = sqrt(DX)

    np.save('./Data/eSol_oneD_mean.npy', mean)
    np.save('./Data/eSol_oneD_std.npy', std)


if __name__ == "__main__":
	get_matrix()
    # cal_mean_std()
