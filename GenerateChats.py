import io
import os
import sys
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Input, Dropout, Dense, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import random

def load_chat():
    '''Loads WhatsApp chat logs from current directory'''

    direc = os.getcwd()
    files = [os.path.join(direc, f) for f in os.listdir(direc) if 'WhatsApp Chat' in f]
    chat = []

    ## Append all chats to one list
    for file in files:
        with io.open(file, encoding="utf-8") as TextFile:
            chat.extend(TextFile.readlines())

    return chat

def preProcess(chat):
    '''Takes a list of messages and removes all timestamps & media messages.
       Returns a list of messages with all special characters and emoji removed'''

    ## Remove media message remnants & timestamps
    nomedia = [message[20:].lower() for message in chat if "<Media omitted>" not in message]

    m = len(nomedia)

    filter = list(' eaotinslrhdmcy\nu:gwbkfpj\'xv?.054,7912/"3!z-q86+£=(_)&*%@#;$\\~^<[]>')

    processed =[]

    for i in range(m):
        filtered = ""
        for char in nomedia[i]:
            if char in filter:
                filtered += char

        ## Omit completely filtered (blank) messages
        if ": \n" not in filtered:
            processed.append(filtered.strip(" -"))

    return processed

def RNNModel(maxLen, vocabSize):

    Inputs = Input(shape=(maxLen,), dtype="float")

    Embeddings = Embedding(vocabSize, 500, dtype="float", input_length=maxLen, trainable=True)
    Embeddings = Embeddings(Inputs)

    X = LSTM(512, input_shape=(maxLen, 500), return_sequences=True)(Embeddings)
    X = LSTM(512, return_sequences=True)(X)
    X = LSTM(512, return_sequences=False)(X)
    X = Dense(1024, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(vocabSize, activation="softmax")(X)

    model = Model(inputs=Inputs, outputs=X)

    return model

class chatGenerator(Callback):
    '''Callback to write a chat dialogue on every epoch end'''

    def __init__(self, Tokenize, unTokenize, wordCorpus, maxLen):
        self.tok = Tokenize
        self.utok = unTokenize
        self.wc = wordCorpus
        self.ml = maxLen

    def on_epoch_end(self, epoch, logs=None):
        print('\n-----------------------------------------------------------------------------------------------\n')

        seed = random.randint(0, len(self.wc) - self.ml - 1)
        generated = []
        sentence = self.wc[seed: seed + self.ml]
        generated += sentence
        sys.stdout.write(" ".join(generated) + " ")

        for i in range(150):
            X_pred = np.zeros([1, self.ml])
            for t in range(self.ml):
                X_pred[0, t] = self.tok[sentence[t]]

            preds = self.model.predict(X_pred, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')
            preds = preds / np.sum(preds)

            next_index = np.argmax(np.random.multinomial(1, preds, 1))
            next_word = self.utok[next_index]

            generated.append(next_word)
            sentence = sentence[1:]
            sentence.append(next_word)
            sys.stdout.write(next_word + " ")
            sys.stdout.flush()

        print('\n-----------------------------------------------------------------------------------------------\n')

def train_on_chat(chat):
    ''' - Transforms a list of chat messages into sequences of one-hot word vectors, length 7 words, and stores them in X
        - The trailing words for each sequence are stored in Y
        - Trains the LSTM model described in RNNModel() on X to predict Y for each message
        - Generates a sample chat on each epoch end '''

    n = len(chat)
    print("\nTotal Messages:", n)
    seed = int(np.random.rand() * n)
    print("Here's a random entry:", chat[seed])

    maxLen = 7
    corpus = []

    chatSplit = [s.replace("\n", " \n").replace("!", " !").replace("?", " ?").replace(":", " :").replace(".", " .").replace(",", " ,").split(" ") for s in chat]

    for i in range(n):
        corpus += chatSplit[i]  # Concatenate all messages together

    wordCorpus = corpus.copy()

    T_words = sorted(set(corpus))
    " ".join(corpus)
    Tokenize = dict((c, i) for i, c in enumerate(T_words))
    unTokenize = dict((i, c) for i, c in enumerate(T_words))

    vocabSize = len(Tokenize.keys()) + 1

    T_sequences = []
    next_words = []
    for i in range(0, len(wordCorpus) - maxLen, 11):  # Split corpus into sequences of maxLen words
        T_sequences.append(wordCorpus[i: i + maxLen])
        next_words.append(wordCorpus[i + maxLen])

    m = len(T_sequences)
    if m == 0: raise Exception('No messages present in chatlog')

    X = np.zeros([m, maxLen])
    Y = np.zeros([m, vocabSize])

    for i, sequence in enumerate(T_sequences):
        for t, word in enumerate(sequence):
            X[i, t] = Tokenize[word]
        Y[i, Tokenize[next_words[i]]] = 1

    print("Using {} sequences for training...".format(m))
    model = RNNModel(maxLen, vocabSize)
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.0005, clipvalue=1), metrics=['accuracy'])
    callback = chatGenerator(Tokenize, unTokenize, wordCorpus, maxLen)

    model.fit(X, Y, epochs=10, batch_size=32, shuffle=True, callbacks=[callback])

if __name__ == '__main__':
    chat = load_chat()
    chat = preProcess(chat)
    train_on_chat(chat)