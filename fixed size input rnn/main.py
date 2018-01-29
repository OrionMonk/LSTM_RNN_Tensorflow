import tensorflow as tf 
from lib import load, summarize
import numpy as np

def run():
	# init
	n_hidden = 16
	n_epochs = 1000
	batch_size = 128
	seq_len = 10
	feature_size = 1
	label_dim = 4

	# loading model
	train_features, train_labels, test_features, test_labels = load.loadf(seq_len=seq_len, label_dim = label_dim)
	train_features = train_features[:(train_features.shape[0]//batch_size)*batch_size]
	train_labels = train_labels[:(train_labels.shape[0]//batch_size)*batch_size]
	
	# variables
	ph_input = tf.placeholder(tf.float32, shape = [None, seq_len, feature_size])
	ph_labels = tf.placeholder(tf.float32, shape = [None, label_dim])

	# rnn 
	lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
	outputs, state = tf.nn.dynamic_rnn(lstm_cell, ph_input, dtype = tf.float32)

	''' Retrieving last output from the sequence of outputs,
	 outputs is of shape batch_size x sequence_length x feature_size ''' 
	outputs = tf.transpose(outputs, [1,0,2])
	last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

	# output layer
	output = tf.layers.dense(inputs = last_output, units = label_dim, kernel_initializer = tf.truncated_normal_initializer())
	prediction = tf.nn.softmax(output)

	# loss minimization
	loss = tf.nn.softmax_cross_entropy_with_logits(labels = ph_labels, logits = prediction)
	minimize = tf.train.AdamOptimizer(0.005).minimize(loss)

	error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(prediction, 1), tf.argmax(ph_labels, 1)), tf.float32))

	# training phase
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			# shuffling and preprocessing data
			train_features, train_labels = load.shuffle(train_features, train_labels, seq_len = seq_len, label_dim = label_dim)
			train_features = train_features.reshape([-1, batch_size, seq_len, feature_size])
			train_labels = train_labels.reshape([-1, batch_size, label_dim])

			# print(train_features.shape, train_labels.shape, "\n",train_features[0][0], train_labels[0][0])

			# epoch loop
			for batch_id in range(train_features.shape[0]):
				sess.run([minimize], feed_dict = {
					ph_input : train_features[batch_id],
					ph_labels : train_labels[batch_id]
					})

			curr_error = sess.run(error, feed_dict = {
					ph_input : test_features,
					ph_labels : test_labels,
					}) 

			if not epoch%1:
				summarize.summarize_epoch(epoch, n_epochs, error = curr_error)

		test_prediction = sess.run(prediction, {
			ph_input:test_features[:20],
			ph_labels:test_labels[:20]
			})
		# print(test_features[:20])
		print("Truth Labels: \t\t",np.argmax(test_labels[:20], axis = 1))
		print("Predicted Labels: \t",np.argmax(test_prediction, axis = 1))

if __name__ == "__main__":
	run()