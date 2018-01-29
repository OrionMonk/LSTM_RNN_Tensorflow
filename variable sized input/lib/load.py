import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# get one hot vectors for each character
def one_hot(num):
	vocab_size = 26
	return np.eye(vocab_size)[num-97]

def loadf(file_name, max_sqlen, batch_size):
	# read data file
	file = open(file_name,"r")
	lines = csv.reader(file)

	# collect names and labels and sequence lengths
	names, labels, lengths = [],[], []
	for row in lines:
		# print(row[0], row[1])
		if "boy" in row[1]:
			names.append([one_hot(ord(char)) for char in row[0].lower()])
			labels.append([1., 0.])
			lengths.append([len(row[0])])
		elif "girl" in row[1]:
			names.append([one_hot(ord(char)) for char in row[0].lower()])
			labels.append([0., 1.])
			lengths.append([len(row[0])])

	names = np.array(names)
	# round off data limit to fit batch size
	data_limit = batch_size*(names.shape[0]//batch_size)

	# pad end of sequences with zeros and limit the data size
	names = pad_sequences(names, padding = 'post', maxlen = max_sqlen)[:data_limit]
	labels = np.array(labels)[:data_limit]
	lengths = np.array(lengths)[:data_limit]

	# reshape to batched data
	names = names.reshape([-1, batch_size, names.shape[1], names.shape[2]])
	labels = labels.reshape([-1, batch_size, labels.shape[1]])
	lengths = lengths.reshape([-1, batch_size])

	return names, labels, lengths

def get_batches(names, labels, sqlen, batch_size, shuffled = True):
	max_sqlen = names.shape[2]
	feature_size = names.shape[3]

	# reshape for proper combination
	names = names.reshape([-1, max_sqlen*feature_size])
	labels = labels.reshape([-1, labels.shape[2]])
	sqlen = sqlen.reshape([-1, 1])

	# shuffle data after combining
	combined = np.append(names, labels, axis = 1)
	combined = np.append(combined, sqlen, axis = 1)
	np.random.shuffle(combined)

	# retrieve individual values
	names = combined[:,:max_sqlen*feature_size]
	labels = combined[:,max_sqlen*feature_size:max_sqlen*feature_size+labels.shape[1]]
	sqlen = combined[:,max_sqlen*feature_size+labels.shape[1]:]

	# reshape for batched data
	names = names.reshape([-1, batch_size, max_sqlen, feature_size])
	labels = labels.reshape([-1, batch_size, labels.shape[1]])
	sqlen = sqlen.reshape([-1, batch_size])

	return names, labels, sqlen