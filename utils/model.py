import tensorflow as tf

def length(mask):
    mask = tf.cast(mask, tf.int32)
    length = tf.reduce_sum(mask, axis=1)
    return (length)


def prepro_for_softmax(logits, mask):
    # Make the indexes of the mask values of 1 and indexes of non mask 0
    new_mask = tf.subtract(tf.constant(1.0), tf.cast(mask, tf.float32))
    mask_value = tf.multiply(new_mask, tf.constant(-1e9))
    masked_logits = tf.where(mask, logits, mask_value)
    return masked_logits


def logits_helper(context, max_context_length):
    d = context.get_shape().as_list()[-1]
    context = tf.reshape(context, shape=[-1, d])
    W = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    pred = tf.matmul(context, W)
    pred = tf.reshape(pred, shape=[-1, max_context_length])
    return pred


def get_optimizer(opt, learning_rate):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        assert False
    return optfn


def BiLSTM(inputs, masks, size, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(size, reuse=reuse)
    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(size, reuse=reuse)
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout)

    sequence_length = length(masks)

    (output_fw, output_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                               cell_bw,
                                                                                               inputs=inputs,
                                                                                               initial_state_fw=initial_state_fw,
                                                                                               initial_state_bw=initial_state_bw,
                                                                                               dtype=tf.float32,
                                                                                               sequence_length=sequence_length)
    output_concat = tf.concat([output_fw, output_bw], 2)
    return (output_concat, (final_state_fw, final_state_bw))
