import io
import os
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Input, Dropout, Dense, Embedding, Activation
from keras.optimizers import RMSprop
from generator import ChatGenerator


class NoFriendsError(Exception):
    """Raised when the message log is empty after filtering."""

    def __init__(self, message=None):
        self.message = message or "Message list is empty."

    def __str__(self):
        return self.message


def load_chat():
    """Loads WhatsApp chat logs from current directory."""

    direc = os.getcwd()
    files = [os.path.join(direc, f) for f in os.listdir(direc) if "WhatsApp Chat" in f]
    chat = []

    # append all chats to one list
    for file in files:
        with io.open(file, encoding="utf-8") as TextFile:
            chat.extend(TextFile.readlines())

    if len(chat) == 0:
        raise NoFriendsError

    return chat


def pre_process(chat):
    """Takes a list of messages and removes all timestamps & media messages.
       Returns a list of messages with all special characters and emoji removed"""

    # remove media message remnants & timestamps
    nomedia = [
        message[20:].lower() for message in chat if "<Media omitted>" not in message
    ]

    m = len(nomedia)

    filter = list(
        " eaotinslrhdmcy\nu:gwbkfpj'xv?.054,7912/\"3!z-q86+£=(_)&*%@#;$\\~^<[]>"
    )

    processed = []

    for i in range(m):
        filtered = ""
        for char in nomedia[i]:
            if char in filter:
                filtered += char

        # omit completely filtered (blank) messages
        if ": \n" not in filtered and len(filtered) > 1:
            processed.append(filtered.strip(" -"))

    return processed


def rnn_model(maxLen, vocabSize):

    Inputs = Input(shape=(maxLen,), dtype="float")

    Embeddings = Embedding(
        vocabSize, 512, dtype="float", input_length=maxLen, trainable=True
    )
    Embeddings = Embeddings(Inputs)

    X = LSTM(256, input_shape=(maxLen, 512), return_sequences=True)(Embeddings)
    X = LSTM(128, return_sequences=False)(X)
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(256)(X)
    X = Dense(vocabSize)(X)
    X = Activation("softmax")(X)

    model = Model(inputs=Inputs, outputs=X)

    return model


def train_on_chat(chat):
    """ - Transforms a list of chat messages into sequences of one-hot word vectors, length 7 words, and stores them in X
        - The trailing words for each sequence are stored in Y
        - Trains the LSTM model described in RNNModel() on X to predict Y for each message
        - Generates a sample chat on each epoch end """

    n = len(chat)
    print("\nTotal Messages:", n)
    seed = int(np.random.rand() * n)
    print("Here's a random entry:", chat[seed])

    max_len = 7
    corpus = []

    chat_split = [
        s.replace("\n", " \n")
        .replace("!", " !")
        .replace("?", " ?")
        .replace(":", " :")
        .replace(".", " .")
        .replace(",", " ,")
        .split(" ")
        for s in chat
    ]

    for i in range(n):
        corpus += chat_split[i]  # Concatenate all messages together

    word_corpus = corpus.copy()

    t_words = sorted(set(corpus))
    " ".join(corpus)
    tokenize = dict((c, i) for i, c in enumerate(t_words))
    untokenize = dict((i, c) for i, c in enumerate(t_words))

    vocab_size = len(tokenize.keys())

    t_sequences = []
    next_words = []
    for i in range(
        0, len(word_corpus) - max_len, 3
    ):  # Split corpus into sequences of maxLen words
        t_sequences.append(word_corpus[i : i + max_len])
        next_words.append(word_corpus[i + max_len])

    m = len(t_sequences)
    if m == 0:
        raise NoFriendsError

    X = np.zeros([m, max_len])
    Y = np.zeros([m, vocab_size])

    for i, sequence in enumerate(t_sequences):
        for t, word in enumerate(sequence):
            X[i, t] = tokenize[word]
        Y[i, tokenize[next_words[i]]] = 1

    print("Using {} sequences for training...".format(m))
    model = rnn_model(max_len, vocab_size)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(lr=0.0005, clipvalue=1),
        metrics=["accuracy"],
    )
    callback = ChatGenerator(tokenize, untokenize, word_corpus, max_len)

    model.fit(X, Y, epochs=20, batch_size=32, shuffle=True, callbacks=[callback])
    return 1


if __name__ == "__main__":
    chat = load_chat()
    chat = pre_process(chat)
    train_on_chat(chat)
