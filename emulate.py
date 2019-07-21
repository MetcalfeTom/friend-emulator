import numpy as np
from keras.optimizers import RMSprop
from friend_emulator.callback import ChatGenerator
from friend_emulator.model import rnn_model
from friend_emulator.data import DataFriend


def train_on_chat(chat):
    """ - Transforms a list of chat messages into sequences of one-hot word vectors, length 7 words, and stores them in X
        - The trailing words for each sequence are stored in Y
        - Trains the LSTM model described in RNNModel() on X to predict Y for each message
        - Generates a sample chat on each epoch end """

    n = len(chat)
    print("\nTotal Messages:", n)
    seed = int(np.random.rand() * n)
    print("Here's a random entry:", chat[seed].get("text"))

    max_len = 7
    corpus = []

    # TODO: use spaCy tokenizer
    chat_split = ["{}: {}".format(s.get("user"), s.get("text")) for s in chat]
    chat_split = [
        s.replace("\n", " \n")
        .replace("!", " !")
        .replace("?", " ?")
        .replace(":", " :")
        .replace(".", " .")
        .replace(",", " ,")
        .split(" ")
        for s in chat_split
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


if __name__ == "__main__":
    friend = DataFriend()
    train_on_chat(friend.messages)
