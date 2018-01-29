import tensorflow as tf 
from lib import load, summarize
import numpy as np


def run(train = True):
	# init
	n_hidden = 32
	n_epochs = 5000
	batch_size = 256
	seq_len = 10
	feature_size = 1
	label_dim = 8

	# variables
	ph_input = tf.placeholder(tf.float32, [None, seq_len, feature_size])
	ph_labels = tf.placeholder(tf.float32, [None, label_dim])
	ph_cell_state = tf.placeholder(tf.float32, [None, n_hidden])
	ph_hidden_state = tf.placeholder(tf.float32, [None, n_hidden])
	init_state = tf.nn.rnn_cell.LSTMStateTuple(ph_cell_state, ph_hidden_state)

	# rnn 
	lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, ph_input, initial_state = init_state, dtype = tf.float32)

	''' Retrieving last output from the sequence of outputs,
	 outputs is of shape batch_size x sequence_length x feature_size ''' 
	outputs = tf.transpose(outputs, [1,0,2])
	last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

	# output layer
	output = tf.layers.dense(inputs = last_output, units = label_dim, kernel_initializer = tf.truncated_normal_initializer())
	prediction = tf.nn.softmax(output)

	# loss minimization
	loss = tf.nn.softmax_cross_entropy_with_logits(labels = ph_labels, logits = prediction)
	minimize = tf.train.AdamOptimizer(0.001).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ph_labels, 1)), tf.float32))

	# training phase
	with tf.Session() as sess:
		saver = tf.train.Saver()
		# sess.run(tf.global_variables_initializer())
		saver.restore(sess, 'models/rnn.ckpt')

		if train:
			for epoch in range(n_epochs):
				# loading train data each epoch
				features, labels = load.loadf(seq_len, label_dim, feature_size, batch_size)

				# initialise hidden and cell states with zeroes
				_curr_cell_state, _curr_hidden_state = np.zeros((batch_size, n_hidden)), np.zeros((batch_size, n_hidden))

				_curr_acc = 0
				# epoch train loop
				for batch_id in range(features.shape[0]):
					_acc, _, _curr_state = sess.run([accuracy, minimize, final_state], feed_dict = {
						ph_input : features[batch_id],
						ph_labels : labels[batch_id],
						ph_cell_state:_curr_cell_state,
						ph_hidden_state:_curr_hidden_state
						})

					_curr_acc += _acc
					_curr_cell_state, _curr_hidden_state = _curr_state

				if not epoch%1:
					summarize.summarize_epoch(epoch, n_epochs, error = _curr_acc/features.shape[0])
					saver.save(sess, 'models/rnn.ckpt')

		test_sample = np.array([
			[[[1], [0], [0], [0], [1], [0], [0], [0], [0], [1]]],
			[[[1], [0], [1], [1], [1], [0], [1], [0], [0], [1]]],
			[[[0], [0], [0], [0], [1], [0], [0], [0], [0], [1]]],
			[[[1], [0], [1], [1], [1], [1], [1], [0], [1], [1]]],
			])

		print("Test Input: ")
		for box in test_sample:
			for item in box[0]:
				print(item[0], end = "\t")
			print()

		print("Predictions")
		_curr_cell_state, _curr_hidden_state = np.zeros((test_sample.shape[1], n_hidden)), np.zeros((test_sample.shape[1], n_hidden))
		for batch_id in range(test_sample.shape[0]):
			curr_prediction, _curr_state = sess.run([prediction, final_state], {
				ph_input:test_sample[batch_id],
				ph_cell_state:_curr_cell_state,
				ph_hidden_state:_curr_hidden_state
				})
			_curr_cell_state, _curr_hidden_state = _curr_state
			print(np.around(curr_prediction), " -> ",np.argmax(curr_prediction, 1)[0])

if __name__ == "__main__":
	# set fixed random seed for tensorflow
	tf.set_random_seed(10)

	run()