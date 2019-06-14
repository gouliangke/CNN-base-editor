#generate a look up dictionary
import numpy as np
import itertools   
from itertools import permutations 
import os, sys
from re import finditer
from collections import defaultdict


def count_kmers(read, k, features):
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = read[i:i+k]
        features[index[kmer]] += 1
    return features


#Build the index for bases to the output matrix column 

perms = ['A','T','C','G']

tmp = [x for x in itertools.product('ACGT', repeat=2)]
tmp = tmp + [x for x in itertools.product('ACGT', repeat=3)]
tmp = tmp + [x for x in itertools.product('ACGT', repeat=4)]
tmp = tmp + [x for x in itertools.product('ACGT', repeat=5)]

perms = perms + [''.join(w) for w in tmp]

initial = range(len(perms))
index = dict(zip(perms,initial))

##load data 
def load_data(fp='data_30bp_72.txt'):
    X = []
    with open(fp, 'r') as f:
        for line in f:
            seq, resp = line.strip().split()
            X.append( seq)
            #Y.append(float(resp))
    X = np.array(X)
    return X

X = load_data()

features = np.zeros((len(X), len(perms)), dtype = int)

for j in range(len(X)):
    for k in range(1,5):
        features[j,:] = count_kmers(X[j], k, features[j,:])

np.save('features_5mer', features)



