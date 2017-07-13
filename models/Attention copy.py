import logging
from utils.general import batches, Progbar, pad_sequences, get_random_samples, find_best_span, save_graphs
from utils.eval import evaluate
import numpy as np
import tensorflow as tf
from os.path import join as pjoin

logging.basicConfig(level=logging.INFO)


def length(mask):
    mask = tf.cast(mask, tf.int32)
    length = tf.reduce_sum(mask, axis=1)
    return (length)


def prepro_for_softmax(logits, mask):
    # Make the indexes of the mask values of 1 and indexes of non mask 0
    new_mask = tf.subtract(tf.constant(1.0), tf.cast(mask, tf.float32))
    mask_value = tf.multiply(new_mask, tf.constant(-1e9))
    logits = tf.where(mask, logits, mask_value)
    return (logits)


def logits_helper(context, max_context_length):
    d = context.get_shape().as_list()[-1]
    context = tf.reshape(context, shape=[-1, d])
    W = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    pred = tf.matmul(context, W)
    pred = tf.reshape(pred, shape=[-1, max_context_length])
    return (pred)


def get_optimizer(opt, learning_rate):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        assert (False)
    return (optfn)


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
    def calculate(self, Hq, Hc, max_question_length, max_context_length, question_mask, context_mask, is_train,
                  dropout):
        # (BS, MCL, MQL, HS * 2)
        d = Hq.get_shape().as_list()[-1]
        logging.debug("d is: {}".format(d))

        # (BS, MPL, MQL)
        interaction_weights = tf.get_variable("W_interaction", shape=[d, d])
        Hc_W = tf.reshape(tf.reshape(Hc, shape=[-1, d]) @ interaction_weights,
                          shape=[-1, max_context_length, d])

        # (BS, MPL, HS * 2) @ (BS, HS * 2, MCL) -> (BS ,MCL, MQL)
        score = Hc_W @ tf.transpose(Hq, [0, 2, 1])

        # Create mask (BS, MPL) -> (BS, MPL, 1) -> (BS, MPL, MQL)
        context_mask_aug = tf.tile(tf.expand_dims(context_mask, 2), [1, 1, max_question_length])
        question_mask_aug = tf.tile(tf.expand_dims(question_mask, 1), [1, max_context_length, 1])
        mask_aug = context_mask_aug & question_mask_aug

        score_prepro = prepro_for_softmax(score, mask_aug)  # adds around ~2% to EM

        # (BS, MPL, MQL)
        alignment_weights = tf.nn.softmax(score_prepro)

        # (BS, MPL, MQL) @ (BS, MQL, HS * 2) -> (BS, MPL, HS * 2)
        context_aware = tf.matmul(alignment_weights, Hq)

        concat_hidden = tf.concat([context_aware, Hc], axis=2)
        concat_hidden = tf.cond(is_train, lambda: tf.nn.dropout(concat_hidden, dropout), lambda: concat_hidden)

        # (HS * 4, HS * 2)
        Ws = tf.get_variable("Ws", shape=[d * 2, d])
        attention = tf.nn.tanh(tf.reshape(tf.reshape(concat_hidden, [-1, d * 2]) @ Ws,
                                          [-1, max_context_length, d]))
        return (attention)


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs, mask, max_input_length, dropout):
        with tf.variable_scope("m1"):
            m1, _ = BiLSTM(inputs, mask, self.output_size, dropout=dropout)

        with tf.variable_scope("m2"):
            m2, _ = BiLSTM(m1, mask, self.output_size, dropout=dropout)

        with tf.variable_scope("start"):
            start = logits_helper(m2, max_input_length)
            start = prepro_for_softmax(start, mask)

        with tf.variable_scope("end"):
            end = logits_helper(m2, max_input_length)
            end = prepro_for_softmax(end, mask)

        return start, end


class LuongAttention(object):
    def __init__(self, result_saver, embeddings, config):

        self.result_saver = result_saver
        self.embeddings = embeddings
        self.config = config
        self.encoder = Encoder(config.hidden_size)
        self.decoder = Decoder(config.hidden_size)
        self.attention = Attention()

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_span_start_placeholder = tf.placeholder(tf.int32)
        self.answer_span_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # ==== assemble pieces ====
        with tf.variable_scope("Attention", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            logging.debug("question_embeddings shape: {}".format(self.question_embeddings.get_shape()))
            logging.debug("context_embeddings shape: {}".format(self.context_embeddings.get_shape()))
            self.preds_op = self.setup_system()
            self.loss_op = self.setup_loss()

            # ==== set up training/updating procedure ====
            self.train_op = self.setup_train()

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

    def setup_system(self):

        # First we set up the encoding 

        logging.info(("-" * 10, "ENCODING ", "-" * 10))
        with tf.variable_scope("q"):
            Hq, (q_final_state_fw, q_final_state_bw) = self.encoder.encode(self.question_embeddings,
                                                                           self.question_mask_placeholder,
                                                                           dropout=self.dropout_placeholder)

            if self.config.share_encoder_weights:
                Hc, (c_final_state_fw,  c_final_state_bw) = self.encoder.encode(self.context_embeddings,
                                                                               self.context_mask_placeholder,
                                                                               initial_state_fw=q_final_state_fw,
                                                                               initial_state_bw=q_final_state_bw,
                                                                               dropout=self.dropout_placeholder,
                                                                               reuse=True)
            else:
                with tf.variable_scope("c"):
                    Hc, (c_final_state_fw, c_final_state_bw) = self.encoder.encode(self.context_embeddings,
                                                                                   self.context_mask_placeholder,
                                                                                   initial_state_fw=q_final_state_fw,
                                                                                   initial_state_bw=q_final_state_bw,
                                                                                   dropout=self.dropout_placeholder)

        # Now setup the attention module
        with tf.variable_scope("attention"):
            attention = self.attention.calculate(Hq, Hc, self.max_question_length_placeholder,
                                                 self.max_context_length_placeholder,
                                                 self.question_mask_placeholder, self.context_mask_placeholder,
                                                 is_train=(self.dropout_placeholder < 1.0),
                                                 dropout=self.dropout_placeholder)

        logging.info(("-" * 10, " DECODING ", "-" * 10))
        with tf.variable_scope("decoding"):
            start, end = self.decoder.decode(attention, self.context_mask_placeholder,
                                             self.max_context_length_placeholder, self.dropout_placeholder)
        return start, end

    def setup_loss(self):
        with tf.variable_scope("loss"):
            answer_span_start_one_hot = tf.one_hot(self.answer_span_start_placeholder, self.max_context_length_placeholder)
            answer_span_end_one_hot = tf.one_hot(self.answer_span_end_placeholder, self.max_context_length_placeholder)
            logging.info("answer_span_start_one_hot.get_shape() {}".format(answer_span_start_one_hot.get_shape()))
            logging.info("answer_span_end_one_hot.get_shape() {}".format(answer_span_end_one_hot.get_shape()))

            start, end = self.preds_op
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=start, labels=answer_span_start_one_hot))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end, labels=answer_span_end_one_hot))
            loss = loss1 + loss2
        return loss

    def setup_train(self):
        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, variables)
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

    def optimize(self, session, batch):
        context = batch["context"]
        question = batch["question"]
        answer_span_start = batch["answer_span_start"]
        answer_span_end = batch["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end)
        output_feed = [self.train_op, self.loss_op]

        outputs = session.run(output_feed, input_feed)

        return (outputs)

    def test(self, session, val):
        context = val["context"]
        question = val["question"]
        answer_span_start = val["answer_span_start"]
        answer_span_end = val["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end, is_train=False)
        output_feed = self.loss_op

        outputs = session.run(output_feed, input_feed)

        return (outputs)

    def decode(self, session, batch):
        context = batch["context"]
        question = batch["question"]
        answer_span_start = batch["answer_span_start"]
        answer_span_end = batch["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end, is_train=False)
        output_feed = self.preds_op

        start, end = session.run(output_feed, input_feed)

        return (start, end)

    def answer(self, session, data, use_best_span):

        start, end = self.decode(session, data)

        # logging.debug("start shape: {}".format(start.shape))
        # logging.debug("end shape: {}".format(end.shape))

        if use_best_span:
            start_index, end_index = find_best_span(start, end)
        else:
            start_index = np.argmax(start, axis=1)
            end_index = np.argmax(end, axis=1)

        # logging.debug("start_index: {}".format(start_index))
        # logging.debug("end_index: {}".format(end_index))

        return (start_index, end_index)

    def validate(self, sess, val):
        valid_cost = self.test(sess, val)

        return (valid_cost)

    def _get_sentences_from_indicies(self, data, start_index, end_index):
        answer_word_pred = []
        answer_word_truth = []
        word_context = data["word_context"]
        answer_span_start = data["answer_span_start"]
        answer_span_end = data["answer_span_end"]

        for span, context in zip(zip(start_index, end_index), word_context):
            prediction = " ".join(context.split()[span[0]:span[1] + 1])
            #         print(prediction)
            answer_word_pred.append(prediction)

        for span, context in zip(zip(answer_span_start, answer_span_end), word_context):
            truth = " ".join(context.split()[span[0]:span[1] + 1])
            #         print(truth)
            answer_word_truth.append(truth)

        return (answer_word_pred, answer_word_truth)

    def predict_for_batch(self, session, data, use_best_span):
        batch_num = int(np.ceil(len(data) * 1.0 / self.config.batch_size))
        start_indicies = []
        end_indicies = []
        for batch in batches(data, is_train=False, batch_size=self.config.batch_size, shuffle=False):
            # logging.debug("batch is: {}".format(batch))
            start_index, end_index = self.answer(session, batch, use_best_span)
            start_indicies.extend(start_index)
            end_indicies.extend(end_index)
        # logging.debug("start_indicies: {}".format(start_indicies))
        # logging.debug("end_indicies: {}".format(end_indicies))
        return (start_indicies, end_indicies)

    def evaluate_answer(self, session, data, num_samples, use_best_span):

        # Now we whether finding the best span improves the score
        start_indicies, end_indicies = self.predict_for_batch(session, data, use_best_span)
        pred_answer, truth_answer = self._get_sentences_from_indicies(data, start_indicies, end_indicies)
        result = evaluate(pred_answer, truth_answer)

        f1 = result["f1"]
        EM = result["EM"]

        return f1, EM

    def run_epoch(self, session, train, val, log=False):
        num_samples = len(train["context"])
        num_batches = int(np.ceil(num_samples) * 1.0 / self.config.batch_size)
        progress = Progbar(target=num_batches)
        for i, train_batch in enumerate(
                batches(train, is_train=True, batch_size=self.config.batch_size, window_size=self.config.window_size)):
            _, loss = self.optimize(session, train_batch)
            progress.update(i + 1, [("training loss", loss)])
            self.result_saver.save("losses", loss)

            if i % self.config.eval_num == 0:

                # Randomly get some samples from the dataset
                train_samples = get_random_samples(train, self.config.samples_used_for_evaluation)
                val_samples = get_random_samples(val, self.config.samples_used_for_evaluation)

                # First evaluate on the training set for not using best span
                f1_train, EM_train = self.evaluate_answer(session, train_samples,
                                                          num_samples=self.config.samples_used_for_evaluation,
                                                          use_best_span=False)

                # Then evaluate on the val set
                f1_val, EM_val = self.evaluate_answer(session, val_samples,
                                                      num_samples=self.config.samples_used_for_evaluation,
                                                      use_best_span=False)

                if log:
                    print()
                    print("Not using best span")
                    logging.info("F1: {}, EM: {}, for {} training samples".format(f1_train, EM_train,
                                                                                  self.config.samples_used_for_evaluation))
                    logging.info("F1: {}, EM: {}, for {} validation samples".format(f1_val, EM_val,
                                                                                    self.config.samples_used_for_evaluation))

                # First evaluate on the training set
                f1_train, EM_train = self.evaluate_answer(session, train_samples,
                                                          num_samples=self.config.samples_used_for_evaluation,
                                                          use_best_span=True)

                # Then evaluate on the val set
                f1_val, EM_val = self.evaluate_answer(session, val_samples,
                                                      num_samples=self.config.samples_used_for_evaluation,
                                                      use_best_span=True)

                if log:
                    print()
                    print("Using best span")
                    logging.info("F1: {}, EM: {}, for {} training samples".format(f1_train, EM_train,
                                                                                  self.config.samples_used_for_evaluation))
                    logging.info("F1: {}, EM: {}, for {} validation samples".format(f1_val, EM_val,
                                                                                    self.config.samples_used_for_evaluation))

                self.result_saver.save("f1_train", f1_train)
                self.result_saver.save("EM_train", EM_train)
                self.result_saver.save("f1_val", f1_val)
                self.result_saver.save("EM_val", EM_val)
                batches_trained = i if self.result_saver.is_empty("batch_indicies") \
                    else self.result_saver.get("batch_indicies")[-1] + (i + 1)
                self.result_saver.save("batch_indicies", batches_trained)

                save_graphs(self.result_saver.data, path=self.config.train_dir)
                saver = tf.train.Saver()
                saver.save(session, pjoin(self.config.train_dir, "BATCH-{}".format(batches_trained)))

    def train(self, session, train, val):
        variables = tf.trainable_variables()
        num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in variables])
        logging.info("Number of variables in models: {}".format(num_vars))
        for i in range(self.config.num_epochs):
            self.run_epoch(session, train, val, log=self.config.log)
