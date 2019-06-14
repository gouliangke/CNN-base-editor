import numpy as np

def seq2mat(seq):
	letter_idx = {'A':0, 'C':1, 'G':2, 'T':3}
	mat = np.zeros((len(seq), 4))
	for i in range(len(seq)):
		try:
			j = letter_idx[seq[i]]
			mat[i, j] = 1 
		except:
			mat[i,:] = 0.25
	return mat


def load_data(fp='data_30bp_72.txt'):
	X = []
	Y = []
	with open(fp, 'r') as f:
		for line in f:
			seq, resp = line.strip().split()
			if len(seq)!=30:
				continue
			if resp=='NA':
				resp = -10.
			X.append( seq2mat(seq) )
			#Y.append(float(resp))
			if float(resp)>0:
				Y.append(0)
			else:
				Y.append(1)
	X = np.array(X)
	Y = np.array(Y)
	return X, Y