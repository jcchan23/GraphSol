import os
import numpy as np
from tqdm import tqdm


def read_spotcon(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index < 5:
            continue
        elif index == 5:
            sequence = line.strip()
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


def read_fasta(fname):
    with open(fname,'r') as f:
        sequence = f.read().splitlines()[1]
    return sequence


spotcon = [name.split('.')[0] for name in os.listdir('./spotcon/')]
for name in tqdm(os.listdir('./fasta/')):
    name = name.split('.')[0]
    if name in spotcon:
        matrix = read_spotcon('./spotcon/'+name+'.spotcon')
    else:
        sequence = read_fasta('./fasta/'+name+'.fasta')
        matrix = np.ones((len(sequence), len(sequence)), dtype=float)
        mask1 = np.tril(np.ones((len(sequence), len(sequence)), dtype=float), -3)
        mask2 = np.triu(np.ones((len(sequence), len(sequence)), dtype=float), 3)
        mask = mask1 + mask2
        matrix -= mask
    np.save('./spotcon_all_c/'+name+'.npy', matrix)



