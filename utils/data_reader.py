from os.path import join as pjoin
import numpy as np

PAD_ID = 0


def get_indicies_sorted_by_context_length(data):
    context_lengths = np.array([len(context) for context in data["context"]])
    return np.argsort(context_lengths)


def reindex_dictionary(data, indices):
    for k, v in data.items():
        data[k] = v[indices]
    return data


def map_indices_to_word_for_question(questions, words):
    word_questions = []
    for question in questions:
        word_question = []
        for word_idx in question:
            word_question.append(words[word_idx])
        word_questions.append(" ".join(word_question))
    return np.array(word_questions)


def load_and_preprocess_data(data_dir):
    words = []
    # First load up the lookup array for indices -> words
    with open(pjoin(data_dir, "vocab.dat"), "r") as f:
        for word in f:
            words.append(word.strip("\n"))

    # Load the training data
    train = {}

    train_context = [list(map(int, line.strip().split()))
                     for line in open(pjoin(data_dir, "train.ids.context"))]

    train_question = [list(map(int, line.strip().split()))
                      for line in open(pjoin(data_dir, "train.ids.question"))]

    train_word_context = [line for line in open(pjoin(data_dir, "train.context"))]
    train_answer_span = [list(map(int, line.strip().split()))
                         for line in open(pjoin(data_dir, "train.span"))]
    train["context"] = np.array(train_context)
    train["question"] = np.array(train_question)

    # Now we need to load word_question which isn't provided to us but that is easy to get
    train["word_context"] = np.array(train_word_context)
    train["answer_span_start"] = np.array(train_answer_span)[:, 0]
    train["answer_span_end"] = np.array(train_answer_span)[:, 1]

    train_indicies = get_indicies_sorted_by_context_length(train)
    train = reindex_dictionary(train, train_indicies)
    train["word_question"] = map_indices_to_word_for_question(train["question"], words)

    # Load the val data
    val = {}

    val_context = [list(map(int, line.strip().split()))
                   for line in open(pjoin(data_dir, "val.ids.context"))]
    val_question = [list(map(int, line.strip().split()))
                    for line in open(pjoin(data_dir, "val.ids.question"))]
    val_word_context = [line for line in open(pjoin(data_dir, "val.context"))]
    val_answer_span = [list(map(int, line.strip().split()))
                       for line in open(pjoin(data_dir, "val.span"))]
    val_answer = [line for line in open(pjoin(data_dir, "val.answer"))]

    val["context"] = np.array(val_context)
    val["question"] = np.array(val_question)
    val["word_context"] = np.array(val_word_context)
    val["answer_span_start"] = np.array(val_answer_span)[:, 0]
    val["answer_span_end"] = np.array(val_answer_span)[:, 1]
    val["word_answer"] = np.array(val_answer)

    val_indicies = get_indicies_sorted_by_context_length(val)
    val = reindex_dictionary(val, val_indicies)

    val["word_question"] = map_indices_to_word_for_question(val["question"], words)
    return train, val


def load_word_embeddings(data_dir):
    return np.load(pjoin(data_dir, "glove.trimmed.100.npz"))["glove"].astype(np.float32)


def create_character_embeddings(data_dir, character_embedding_size):
    character_mappings = {}
    character_mappings["<PAD>"] = 0
    character_mappings["<START>"] = 1
    character_mappings["<END>"] = 2
    character_mappings["<UNK>"] = 3
    count = len(character_mappings) - 1
    with open(pjoin(data_dir, "vocab.dat"), "r") as f:
        for line in f:
            for char in line:
                if char not in character_mappings:
                    character_mappings[char] = count
                    count += 1
    character_embedding = np.random.randn(len(character_mappings) * character_embedding_size).astype(np.float32)
    character_embedding = np.reshape(character_embedding, (len(character_mappings), character_embedding_size))
    return character_embedding, character_mappings
