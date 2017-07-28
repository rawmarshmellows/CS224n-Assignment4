import tensorflow as tf
import logging


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


def logits_helper(context, max_context_length, dropout=None):
    d = context.get_shape().as_list()[-1]
    context = tf.reshape(context, shape=[-1, d])
    W = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    pred = tf.matmul(context, W)
    pred = tf.reshape(pred, shape=[-1, max_context_length])
    if dropout is not None:
        pred = tf.nn.dropout(pred, dropout)
    return pred


def get_optimizer(opt, learning_rate):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        assert False
    return optfn


def word_embedding_lookup(lookup_indices, max_length,
                          lookup_embeddings_matrix, embedding_size):
    embeddings = tf.nn.embedding_lookup(lookup_embeddings_matrix, lookup_indices)
    embeddings = tf.reshape(embeddings, shape=[-1, max_length, embedding_size])
    return embeddings


def character_embedding_lookup(lookup_indices, lookup_embeddings_matrix):
    embeddings = tf.nn.embedding_lookup(lookup_embeddings_matrix, lookup_indices)
    return embeddings


def mask_for_character_embeddings(character_embeddings, mask):
    mask_value = tf.cast(tf.zeros_like(mask), tf.float32)
    masked_chars = tf.where(mask, character_embeddings, mask_value)
    return masked_chars


def conv1d(char_embeddings, filter_widths, num_filters, scope=None):
    outs = []
    d = char_embeddings.get_shape().as_list()[-1]
    with tf.variable_scope(scope or "conv1d"):
        for filter_width, num_filter in zip(filter_widths, num_filters):
            with tf.variable_scope("conv_{}".format(filter_width)):
                W = tf.get_variable("W", shape=[1, filter_width, d, num_filter])
                b = tf.get_variable("b", shape=[num_filter])

                out = tf.nn.conv2d(char_embeddings,
                                   W,
                                   strides=[1, 1, 1, 1],
                                   padding="VALID")
                out = tf.nn.relu(out + b)

                out = tf.reduce_max(out, 2)
                outs.append(out)

    return tf.concat(outs, 2)


def biLSTM(inputs, masks, size, initial_state_fw=None,
           initial_state_bw=None, dropout=1.0, reuse=False):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(size, reuse=reuse)
    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(size, reuse=reuse)
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout)

    sequence_length = length(masks)

    (output_fw, output_bw), (final_state_fw, final_state_bw) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                        cell_bw,
                                        inputs=inputs,
                                        initial_state_fw=initial_state_fw,
                                        initial_state_bw=initial_state_bw,
                                        dtype=tf.float32,
                                        sequence_length=sequence_length)

    output_concat = tf.concat([output_fw, output_bw], 2)
    return output_concat, (final_state_fw, final_state_bw)


def highway(inputs, sequence_length, scope=None):
    input_size = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope or "highway"):
        w_H = tf.get_variable("w_H", shape=[input_size, input_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        w_T = tf.get_variable("w_T", shape=[input_size, input_size],
                              initializer=tf.contrib.layers.xavier_initializer())

        logging.debug("w_H name is: {}".format(w_H.get_shape()))
        logging.debug("w_T name is: {}".format(w_T.get_shape()))
        logging.debug("inputs original shape: {}".format(inputs.get_shape()))

        # Inputs is (BS, MCL/MQL, WES+CES)
        inputs = tf.reshape(inputs, [-1, input_size])
        logging.debug("inputs shape after reshape: {}".format(inputs.get_shape()))

        outputs = inputs @ w_H * inputs @ w_T + inputs * (1 - inputs @ w_T)
        logging.debug("outputs shape: {}".format(outputs.get_shape()))
        logging.debug("sequence_length is: {}".format(sequence_length))
        outputs = tf.reshape(outputs, [-1, sequence_length, input_size])

    return outputs
