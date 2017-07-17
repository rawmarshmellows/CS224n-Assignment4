import logging
from utils.general import pad_sequences
from utils.model import prepro_for_softmax, logits_helper, get_optimizer, BiLSTM
from models.model import Model
import tensorflow as tf
from functools import reduce
from operator import mul


logging.basicConfig(level=logging.INFO)


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
        # The character level embeddings
        # if self.config.char_level_embeddings:
        #     char_level_embeddings = self._character_level_embeddings(inputs, filter_sizes, num_filters, dropout)


        # The contextual level embeddings 
        output_concat, (final_state_fw, final_state_bw) = BiLSTM(inputs, masks, self.size, initial_state_fw,
                                                                 initial_state_bw, dropout, reuse)
        logging.debug("output shape: {}".format(output_concat.get_shape()))

        return (output_concat, (final_state_fw, final_state_bw))

        # def _character_level_embeddings(self, inputs, filter_sizes, num_filters, dropout):

        #     input_flattened = tf.reshape(inputs, )


class Attention(object):
    def __init__(self):
        pass

    # TODO: 
    # _flatten and _reconstruct referenced from ??
    def _similarity_matrix(self, Hq, Hc, max_question_length, max_context_length, question_mask, context_mask, is_train,
                           dropout):
        # (BS, MCL, MQL, HS * 2)
        d = Hq.get_shape().as_list()[-1]
        logging.debug("d is: {}".format(d))
        Hc_aug = tf.tile(tf.reshape(Hc, shape=[-1, max_context_length, 1, d]),
                         [1, 1, max_question_length, 1])

        # (BS, MCL, MQL, HS * 2)
        Hq_aug = tf.tile(tf.reshape(Hq, shape=[-1, 1, max_question_length, d]),
                         [1, max_context_length, 1, 1])

        # [(BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2)]
        args = [Hc_aug, Hq_aug, Hc_aug * Hq_aug]

        def _flatten(tensor, keep):
            fixed_shape = tensor.get_shape().as_list()
            start = len(fixed_shape) - keep

            # Calculate (BS * MCL * MQL)
            left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])

            # out_shape is simply HS * 2
            out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]

            # (BS * MCL * MQL, HS * 2)
            flat = tf.reshape(tensor, out_shape)
            return (flat)

        # [(BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2)]
        args_flat = [_flatten(arg, 1) for arg in args]
        args_flat = [tf.cond(is_train, lambda: tf.nn.dropout(arg, dropout), lambda: arg) for arg in args_flat]

        d_concat = d * 3
        W = tf.get_variable("W", shape=[d_concat, 1])
        b = tf.get_variable("b", shape=[1])

        # Calculating a(h, u) = w_s^(t)[h; u; h * u]
        # (BS * MCL * MQL, HS * 6) @ (HS * 6, 1) + (1) -> (BS * MCL * MQL, 1) 
        res = tf.concat(args_flat, 1) @ W + b
        logging.debug("res shape is: {}".format(res.get_shape()))

        # Now we need to reshape it
        def _reconstruct(tensor, ref, keep):
            ref_shape = ref.get_shape().as_list()
            tensor_shape = tensor.get_shape().as_list()
            ref_stop = len(ref_shape) - keep
            tensor_start = len(tensor_shape) - keep

            # [BS, MCL, MQL]
            pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]

            # [1]
            keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
            # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
            # keep_shape = tensor.get_shape().as_list()[-keep:]

            # [BS, MCL, MQL, 1]
            target_shape = pre_shape + keep_shape
            out = tf.reshape(tensor, target_shape)
            out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
            return (out)

        # (BS * MCL * MQL, 1) -> (BS, MCL, MQL)
        similarity_matrix = _reconstruct(res, args[0], 1)
        logging.debug("similiarity_matrix after reconstruct: {}".format(similarity_matrix.get_shape()))
        context_mask_aug = tf.tile(tf.expand_dims(context_mask, 2), [1, 1, max_question_length])
        question_mask_aug = tf.tile(tf.expand_dims(question_mask, 1), [1, max_context_length, 1])

        mask_aug = context_mask_aug & question_mask_aug
        similarity_matrix = prepro_for_softmax(similarity_matrix, mask_aug)
        return (similarity_matrix)

    def calculate(self, Hq, Hc, max_question_length, max_context_length, question_mask, context_mask, is_train,
                  dropout):
        s = self._similarity_matrix(Hq, Hc, max_question_length, max_context_length, question_mask, context_mask,
                                    is_train, dropout)
        # C2Q

        # (BS, MCL, MQL)
        weights_c2q = tf.nn.softmax(s)

        # (BS, MCL, MQL) @ (BS, MQL, HS * 2) -> (BS, MCL, HS * 2)
        query_aware = weights_c2q @ Hq

        # Q2C

        # (BS, MCL, MQL) -> (BS, MCL)
        # We are effectively looking through all the question words j's to some context word i and finding the 
        # maximum of those context words 
        score_q2c = tf.reduce_max(s, axis=-1)

        # (BS, MCL)
        weights_q2c = tf.expand_dims(tf.nn.softmax(score_q2c), -1)

        # (BS, HS)
        context_aware = tf.reduce_sum(tf.multiply(weights_q2c, Hc), axis=1)

        # (BS, MCL, HS * 2)
        context_aware = tf.tile(tf.expand_dims(context_aware, 1), [1, max_context_length, 1])

        # [(BS, MCL, HS * 2), (BS, MCL, HS * 2), (BS, MCL, HS * 2), (BS, MCL, HS * 2)]
        biattention = tf.nn.tanh(tf.concat([Hc, query_aware, Hc * query_aware, Hc * context_aware], 2))

        return (biattention)


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs, mask, max_input_length, dropout):
        with tf.variable_scope("m1"):
            m1, _ = BiLSTM(inputs, mask, self.output_size, dropout=dropout)

        with tf.variable_scope("m2"):
            m2, _ = BiLSTM(m1, mask, self.output_size, dropout=dropout)

        # Original BiDAF implementation
        # with tf.variable_scope("start"):
        #     start = logits_helper(tf.concat([inputs, m1], 2), max_input_length)
        #     start = prepro_for_softmax(start, mask)

        # with tf.variable_scope("end"):
        #     end = logits_helper(tf.concat([inputs, m2], 2), max_input_length)
        #     end = prepro_for_softmax(end, mask)

        # My implementation 1
        with tf.variable_scope("start"):
            start = logits_helper(tf.concat([inputs, m2], 2), max_input_length)
            start = prepro_for_softmax(start, mask)

        with tf.variable_scope("end"):
            end = logits_helper(tf.concat([inputs, m2], 2), max_input_length)
            end = prepro_for_softmax(end, mask)

        # My implementation 2
        # with tf.variable_scope("start"):
        #     start = logits_helper(m2, max_input_length)
        #     start = prepro_for_softmax(start, mask)

        # with tf.variable_scope("end"):
        #     end = logits_helper(m2, max_input_length)
        #     end = prepro_for_softmax(end, mask)


        return (start, end)


class BiDAF(Model):
    def __init__(self, result_saver, embeddings, config):
        self.embeddings = embeddings
        self.config = config
        self.encoder = Encoder(config.hidden_size)
        self.decoder = Decoder(config.hidden_size)
        self.attention = Attention()
        # ==== set up placeholder tokens ========
        self.add_placeholders()

        # ==== assemble pieces ====
        with tf.variable_scope("Attention", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            self.build(config = config, result_saver = result_saver)

    def add_placeholders(self):
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_span_start_placeholder = tf.placeholder(tf.int32)
        self.answer_span_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def setup_embeddings(self):
        with tf.variable_scope("embeddings"):
            if self.config.retrain_embeddings:
                embeddings = tf.get_variable("embeddings", initializer=self.embeddings)
            else:
                embeddings = tf.cast(self.embeddings, dtype=tf.float32)

            question_embeddings = self._embedding_lookup(embeddings, self.question_placeholder,
                                                         self.max_question_length_placeholder)
            context_embeddings = self._embedding_lookup(embeddings, self.context_placeholder,
                                                        self.max_context_length_placeholder)
        return question_embeddings, context_embeddings

    def _embedding_lookup(self, embeddings, indicies, max_length):
        embeddings = tf.nn.embedding_lookup(embeddings, indicies)
        embeddings = tf.reshape(embeddings, shape=[-1, max_length, self.config.embedding_size])
        return embeddings

    def add_preds_op(self):

        # First we set up the encoding

        logging.info(("-" * 10, "ENCODING ", "-" * 10))
        with tf.variable_scope("q"):
            Hq, (q_final_state_fw, q_final_state_bw) = self.encoder.encode(self.question_embeddings,
                                                                           self.question_mask_placeholder,
                                                                           dropout=self.dropout_placeholder)

            if self.config.share_encoder_weights:
                Hc, (c_final_state_fw, q_final_state_fw) = self.encoder.encode(self.context_embeddings,
                                                                               self.context_mask_placeholder,
                                                                               initial_state_fw=q_final_state_fw,
                                                                               initial_state_bw=q_final_state_bw,
                                                                               dropout=self.dropout_placeholder,
                                                                               reuse=True)
            else:
                with tf.variable_scope("c"):
                    Hc, (c_final_state_fw, q_final_state_fw) = self.encoder.encode(self.context_embeddings,
                                                                                   self.context_mask_placeholder,
                                                                                   initial_state_fw=q_final_state_fw,
                                                                                   initial_state_bw=q_final_state_bw,
                                                                                   dropout=self.dropout_placeholder)

        # Now setup the attention module
        with tf.variable_scope("attention"):
            biattention = self.attention.calculate(Hq, Hc, self.max_question_length_placeholder,
                                                   self.max_context_length_placeholder,
                                                   self.question_mask_placeholder, self.context_mask_placeholder,
                                                   is_train=(self.dropout_placeholder < 1.0),
                                                   dropout=self.dropout_placeholder)

        logging.info(("-" * 10, " DECODING ", "-" * 10))
        with tf.variable_scope("decoding"):
            start, end = self.decoder.decode(biattention, self.context_mask_placeholder,
                                             self.max_context_length_placeholder, self.dropout_placeholder)
        return start, end

    def add_loss_op(self, preds):
        with tf.variable_scope("loss"):
            answer_span_start_one_hot = tf.one_hot(self.answer_span_start_placeholder, self.max_context_length_placeholder)
            answer_span_end_one_hot = tf.one_hot(self.answer_span_end_placeholder, self.max_context_length_placeholder)
            logging.info("answer_span_start_one_hot.get_shape() {}".format(answer_span_start_one_hot.get_shape()))
            logging.info("answer_span_end_one_hot.get_shape() {}".format(answer_span_end_one_hot.get_shape()))

            start, end = preds
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=start, labels=answer_span_start_one_hot))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end, labels=answer_span_end_one_hot))
            loss = loss1 + loss2
        return loss

    def add_training_op(self, loss):
        variables = tf.trainable_variables()
        gradients = tf.gradients(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

        if self.config.learning_rate_annealing:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step, 1250, 0.96,
                                                       staircase=False)
            global_step = tf.add(1, global_step)
        else:
            learning_rate = self.config.learning_rate

        optimizer = get_optimizer(self.config.optimizer, learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, variables))

        # For applying EMA for trained parameters

        if self.config.ema_for_weights:
            ema = tf.train.ExponentialMovingAverage(0.999)
            ema_op = ema.apply(variables)

            with tf.control_dependencies([train_op]):
                train_op = tf.group(ema_op)

        return train_op

    def create_feed_dict(self, context, question, answer_span_start_batch=None, answer_span_end_batch=None,
                         is_train=True):

        # logging.debug("len(context): {}".format(len(context))) 
        # logging.debug("len(question): {}".format(len(question)))

        context_batch, context_mask, max_context_length = pad_sequences(context,
                                                                        max_sequence_length=self.config.max_context_length)
        question_batch, question_mask, max_question_length = pad_sequences(question,
                                                                           max_sequence_length=self.config.max_question_length)

        # logging.debug("context_mask: {}".format(len(context_mask)))
        # logging.debug("question_mask: {}".format(len(question_mask)))

        feed_dict = {self.context_placeholder: context_batch,
                     self.context_mask_placeholder: context_mask,
                     self.question_placeholder: question_batch,
                     self.question_mask_placeholder: question_mask,
                     self.max_context_length_placeholder: max_context_length,
                     self.max_question_length_placeholder: max_question_length}

        if is_train:
            feed_dict[self.dropout_placeholder] = self.config.keep_prob
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        if answer_span_start_batch is not None and answer_span_end_batch is not None:
            feed_dict[self.answer_span_start_placeholder] = answer_span_start_batch
            feed_dict[self.answer_span_end_placeholder] = answer_span_end_batch

        return feed_dict


