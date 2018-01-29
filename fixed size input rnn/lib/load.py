import numpy as np

def loadf(seq_len, label_dim, tt_ratio = 0.8):
	# init
	gen = '{0:0'+str(seq_len)+'b}'

	# param is train to test data ratio
	features = [[float(char) for char in gen.format(i)] for i in range(2**seq_len)]
	labels = [np.eye(label_dim)[bin(i).count("1")%label_dim] for i in range(2**seq_len)]

	features, labels = shuffle(np.array(features), np.array(labels), seq_len, label_dim)
	features = features.reshape([-1, features.shape[1], 1])
	upto = int(tt_ratio*features.shape[0])

	return features[:upto], labels[:upto], features[upto:], labels[upto:]

def shuffle(features, labels, seq_len, label_dim):
	features, labels = features.reshape([-1, seq_len]), labels.reshape([-1, label_dim])
	combined = np.append(features, labels, axis = 1)
	np.random.shuffle(combined)
	return combined[:,:features.shape[1]], combined[:,features.shape[1]:]

