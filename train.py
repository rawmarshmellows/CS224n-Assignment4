import os
from os.path import join as pjoin
import datetime
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

from models.BiDAF import BiDAF
from models.Baseline import Baseline
from models.Attention import LuongAttention
from utils.data_reader import load_and_preprocess_data, load_word_embeddings, create_character_embeddings
from utils.result_saver import ResultSaver

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.80, "The probably that a node is kept after the affine transform")
tf.app.flags.DEFINE_float("max_grad_norm", 5., "The maximum grad norm during back-propagation")
tf.app.flags.DEFINE_integer("batch_size", 24, "Number of batches to be used per training batch")
tf.app.flags.DEFINE_integer("eval_num", 250, "Evaluate on validation set for every eval_num batches trained")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Word embedding size")
tf.app.flags.DEFINE_integer("window_size", 3, "Window size for sampling during training")
tf.app.flags.DEFINE_integer("hidden_size", 100, "Hidden size of the RNNs")
tf.app.flags.DEFINE_integer("samples_used_for_evaluation", 500,
                            "Samples to be used at evaluation for every eval_num batches trained")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of Epochs")
tf.app.flags.DEFINE_integer("max_context_length", None, "Maximum length for the context")
tf.app.flags.DEFINE_integer("max_question_length", None, "Maximum length for the question")
tf.app.flags.DEFINE_integer("character_embedding_size", 50, "Character embedding size")
tf.app.flags.DEFINE_integer("max_word_length", None, "Maximum number of characters in a word")
tf.app.flags.DEFINE_integer("bidaf_output_implementation", 1, "BiDAF output layer")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                                          ), "Saved training parameters directory")
tf.app.flags.DEFINE_string("optimizer", "adam", "The optimizer to be used ")
tf.app.flags.DEFINE_string("model", "BiDAF", "Model type")
tf.app.flags.DEFINE_string("filter_widths", "5", "The number of characters to include per filter (separated by ',')")
tf.app.flags.DEFINE_string("num_filters", "100",
                           "The number of filters to be used for each filter height (separated by ',')")
tf.app.flags.DEFINE_boolean("retrain_embeddings", False, "Whether or not to retrain the embeddings")
tf.app.flags.DEFINE_boolean("share_encoder_weights", False, "Whether or not to share the encoder weights")
tf.app.flags.DEFINE_boolean("learning_rate_annealing", False, "Whether or not to anneal the learning rate")
tf.app.flags.DEFINE_boolean("ema_for_weights", True, "Whether or not to use EMA for weights")
tf.app.flags.DEFINE_boolean("log", True, "Whether or not to log the metrics during training")
tf.app.flags.DEFINE_boolean("find_best_span", True, "Whether find the span with the highest probability")

tf.app.flags.DEFINE_boolean("use_character_embeddings", False, "Whether or not to use the character embeddings")
tf.app.flags.DEFINE_boolean("share_character_cnn_weights", True,
                           "Whether or not to share the CNN weights used to find the character embeddings for words")
tf.app.flags.DEFINE_boolean("use_dropout_before_softmax", False, "Whether not to add dropout to the scores of used for the prediction")
tf.app.flags.DEFINE_boolean("use_highway", True, "Whether or not to use a highway network on the inputs for encoder")
tf.app.flags.DEFINE_boolean("shuffle_batches", True, "Whether or not to shuffle the batches during training")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, train_dir):
    if not os.path.exists(train_dir):
        session.run(tf.global_variables_initializer())
        os.makedirs(train_dir, exist_ok=True)

        # Save the config file
        with open(pjoin(FLAGS.train_dir, "config.txt"), "w") as f:
            output = ""
            for k, v in FLAGS.__flags.items():
                output += "{} : {}\n".format(k, v)
            f.write(output)
    else:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(train_dir)
        saver.restore(session, checkpoint.model_checkpoint_path)


def main(_):
    # load the data
    train, val = load_and_preprocess_data(FLAGS.data_dir)

    # load the word matrix
    embeddings = load_word_embeddings(FLAGS.data_dir)
    logging.info("Load word embeddings of size: {}".format(embeddings.shape))

    # create the character matrix
    character_embeddings, character_mappings = create_character_embeddings(FLAGS.data_dir,
                                                                           FLAGS.character_embedding_size)
    logging.info("Created character embeddings of size: {}".format(character_embeddings.shape))

    # TODO: make this a Singleton object??
    # Create the saver helper object
    result_saver = ResultSaver(FLAGS.train_dir)

    # now load the model
    # with tf.device("/cpu:0"):
    # if FLAGS.model == "BiDAF":
    # 	model = BiDAF(result_saver, embeddings, FLAGS)


    if FLAGS.model == "BiDAF":
        model = BiDAF(result_saver, embeddings, character_embeddings, character_mappings, FLAGS)
    elif FLAGS.model == "Baseline":
        model = Baseline(result_saver, embeddings, FLAGS)
    elif FLAGS.model == "LuongAttention":
        model = LuongAttention(result_saver, embeddings, FLAGS)

    logging.info("Start training with hyper parameters:")

    print(vars(FLAGS)["__flags"])

    with tf.Session() as sess:
        initialize_model(sess, FLAGS.train_dir)
        model.train(sess, train, val)


if __name__ == "__main__":
    tf.app.run()
