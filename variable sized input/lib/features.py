import tensorflow as tf 

# get the last valid end state
def get_end_state(output, seq_len):
	# make a list of increasing indices of size seq_len
	seq_idx = tf.range(tf.shape(seq_len)[0])
	seq_idx = tf.reshape(seq_idx, [-1, 1])
	# reshape seq_len from shape [x] to [x, 1]
	seq_len = tf.reshape(seq_len - 1, [-1, 1])
	# append the indices to each element in seq_len along axis 1
	seq_idx = tf.concat([seq_idx, seq_len], axis = 1)
	# return output[:, seq_len] (in short)
	return tf.gather_nd(output, seq_idx)