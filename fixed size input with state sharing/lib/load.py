import numpy as np

def loadf(seq_len, label_dim, feature_size, batch_size):
	# init
	gen = '{0:0'+str(seq_len)+'b}'
	labels = []

	# feature generation
	features = np.array([[float(char) for char in gen.format(i)] for i in range(2**seq_len)])
	np.random.shuffle(features)

	# shape to n_batches x batch_size x seq_len x feature_size
	features = np.reshape(features,[ -1, batch_size, seq_len, feature_size]) 

	# label generation
	counter = np.array([0 for x in range(batch_size)])
	for i in range(features.shape[0]):
		counter = np.remainder(np.add(counter, [str(x).count("1") for x in features[i]]),label_dim)
		labels.append(np.eye(label_dim)[counter])

	# shape is n_batches x batch_size x label_dim
	labels = np.array(labels) 

	return features, labels
