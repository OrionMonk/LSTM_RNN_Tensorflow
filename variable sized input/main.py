import tensorflow as tf 
import numpy as np 
from lib import features, load, summarize
import numpy as np

def run(max_sqlen):
	# init
	state_size = 256
	n_epochs = 10
	batch_size = 64

	# loading training and test data
	names, labels, sqlen = load.loadf("data/train-names.txt",max_sqlen, batch_size)
	test_names, test_labels, test_sqlen = load.loadf("data/test-names.txt", max_sqlen, 1)

	# resizing test data to shape (batch size, max sequence length, feature dimensions)
	feature_size = names.shape[3]
	label_dim = labels.shape[2]
	test_names = test_names.reshape([-1, max_sqlen, feature_size])
	test_labels = test_labels.reshape([-1, label_dim])
	test_sqlen = test_sqlen.reshape([-1])

	# placholder variables
	ph_names = tf.placeholder(tf.float32, [None, max_sqlen, feature_size]) # placeholder fo sequence name
	ph_labels = tf.placeholder(tf.float32, [None, label_dim])# placeholder for boy/girl label
	ph_seqlen = tf.placeholder(tf.int32, [None]) # placeholder for sequence lengths of each name

	# rnn
	lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple = True)
	outputs, states = tf.nn.dynamic_rnn(lstm_cell, ph_names, sequence_length = ph_seqlen, dtype = tf.float32)

	# getting last valid output from the rnn
	last_output = states[1]
	''' The function commented below may also be used for getting the last output 
	instead of the last hidden state value used, in the above function.
	It uses the ph_seqlen placeholder, to get last valid output from the series of 
	output values returned from the dynamic_rnn function. 
	Check lib/features.py for the code. '''
	# last_output = features.get_end_state(outputs, ph_seqlen)

	# losses and minimzer
	prediction = tf.nn.softmax(tf.layers.dense(last_output, units = label_dim, kernel_initializer = tf.truncated_normal_initializer()))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = ph_labels))
	minimizer = tf.train.AdamOptimizer().minimize(loss)

	# check prediction accuracy by comparing average argmax
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(ph_labels, 1)),tf.float32))

	# start session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# learning phase
		for epoch in range(n_epochs):
			# load shuffled batched data
			names, labels, sqlen = load.get_batches(names, labels, sqlen, batch_size)

			# training
			for batch_id in range(names.shape[0]):
				_, _acc = sess.run([minimizer, accuracy], feed_dict = {
					ph_names : names[batch_id],
					ph_labels : labels[batch_id],
					ph_seqlen : sqlen[batch_id]
					})

				# for batch summary
				if not batch_id%10:
					summarize.summarize_batch(batch_id, names.shape[0], _acc)

			# testing on test samples
			pred, _acc = sess.run([prediction, accuracy], feed_dict = {
				ph_names : test_names,
				ph_labels : test_labels,
				ph_seqlen : test_sqlen
				})

			# pick random sample form the test dataset and print results
			random_no = np.random.randint(test_names.shape[0])
			print("\nTest Accuracy:", _acc)
			print("Name:",[chr(np.argmax(x)+97) if np.count_nonzero(x)>0 else 0 for x in test_names[random_no]])
			print("Prediction",pred[random_no], "\nLabels:",test_labels[random_no])
			print()

if __name__=="__main__":
	tf.set_random_seed(9)
	run(max_sqlen = 11)