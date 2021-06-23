import os
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
from utils import load_blosum, load_aaphy7, load_pssm, load_hhm, load_spd33, load_spotcon
from dataset import ProDataset
from model import GraphSol, BATCH_SIZE

from torch.utils.data import DataLoader
from torch.autograd import Variable


def worker(wkdir, name, sequence, blosum_dict, aaphy7_dict):
    # step 3.1: generate the fasta file
    if not os.path.exists(f'{wkdir}/Data/source/{name}.fasta'):
        with open(f'{wkdir}/Data/source/{name}.fasta', 'w') as fw:
            fw.write(f'>{name}' + '\n')
            fw.write(sequence)

    # step 3.2: generate the final 1D features
    if not os.path.exists(f'{wkdir}/Data/generate/{name}_oneD.npy'):
        blosum = np.array([blosum_dict[amino] for amino in sequence])
        pssm = load_pssm(wkdir, name, sequence)
        hhm = load_hhm(wkdir, name, sequence)
        spd33 = load_spd33(wkdir, name, sequence)
        aaphy7 = np.array([aaphy7_dict[amino] for amino in sequence])
        oneD_matrix = np.concatenate([blosum, pssm, hhm, spd33, aaphy7], axis=1)
        np.save(f'{wkdir}/Data/generate/{name}_oneD.npy', oneD_matrix)
    
    # step 3.3: generate the final 2D features
    if not os.path.exists(f'{wkdir}/Data/generate/{name}_twoD.npy'):
        if os.path.exists(f'{wkdir}/Data/source/{name}.spotcon'):
            twoD_matrix = load_spotcon(wkdir, name, sequence)
        else:
            twoD_matrix = np.ones((len(sequence), len(sequence)), dtype=float)
            mask1 = np.tril(np.ones((len(sequence), len(sequence)), dtype=float), -3)
            mask2 = np.triu(np.ones((len(sequence), len(sequence)), dtype=float), 3)
            mask = mask1 + mask2
            twoD_matrix -= mask
        np.save(f'{wkdir}/Data/generate/{name}_twoD.npy', twoD_matrix)


if __name__=='__main__':

    wkdir = '.'

    with open(f'{wkdir}/Data/upload/input.fasta', 'r') as f:
        lines = f.read().strip().split()

    # step 1 load input files
    sequence_dict = dict()
    for line in lines:
        if line[0] == '>':
            name = line.split(' ')[0][1:]
            sequence_dict[name] = ""
        else:
            sequence_dict[name] += line

    # step 2: load common files
    blosum_dict = load_blosum(wkdir)
    aaphy7_dict = load_aaphy7(wkdir)

    # step 3: generate the node and edge features in parallel
    pool = Pool(processes=4)
    for name, sequence in sequence_dict.items():
        pool.apply_async(worker, (wkdir, name, sequence, blosum_dict, aaphy7_dict, ))
    pool.close()
    pool.join()

    # step 4: construct the dataset and data loader
    test_loader = DataLoader(dataset=ProDataset(wkdir, sequence_dict),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # step 5: predict the result
    result_dict = {}
    for model_name in sorted(os.listdir(f'{wkdir}/Model/')):
        model = GraphSol()
        model.load_state_dict(torch.load(f'{wkdir}/Model/{model_name}', map_location=torch.device('cpu')))
        model.eval()

        for data in test_loader:
            name, sequence, node_features, edge_features = data

            node_features = Variable(torch.squeeze(node_features))
            edge_features = Variable(torch.squeeze(edge_features))

            prediction = model(node_features, edge_features)
            prediction = prediction.detach().numpy().tolist()

            if name[0] not in result_dict:
                result_dict[name[0]] = prediction
            else:
                result_dict[name[0]] += prediction
    
    # step 6: store a file
    result_dataframe = pd.DataFrame(columns=['name', 'prediction', 'sequence'])
    for name, sequence in sequence_dict.items():
        result_dataframe = result_dataframe.append({'name': name,
                                                    'prediction': np.nanmean(result_dict[name]),
                                                    'sequence': sequence}, ignore_index=True)
    result_dataframe.to_csv(f'{wkdir}/Result/result.csv', index=None)









