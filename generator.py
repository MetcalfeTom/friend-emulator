import random
from keras.callbacks import Callback
import sys


class ChatGenerator(Callback):
    """Callback to write a chat dialogue on every epoch end"""

    def __init__(self, tokenize, untokenize, word_corpus, max_len):
        self.tok = tokenize
        self.utok = untokenize
        self.wc = word_corpus
        self.ml = max_len

    def on_epoch_end(self, epoch, logs=None):
        print(
            "\n-----------------------------------------------------------------------------------------------\n"
        )

        seed = random.randint(0, len(self.wc) - self.ml - 1)
        generated = []
        sentence = self.wc[seed : seed + self.ml]
        generated += sentence
        sys.stdout.write(" ".join(generated) + " ")

        for i in range(150):
            X_pred = np.zeros([1, self.ml])
            for t in range(self.ml):
                X_pred[0, t] = self.tok[sentence[t]]

            preds = self.model.predict(X_pred, verbose=0)[0]
            preds = np.asarray(preds).astype("float64")
            preds = preds / np.sum(preds)

            next_index = np.argmax(np.random.multinomial(1, preds, 1))
            next_word = self.utok[next_index]

            generated.append(next_word)
            sentence = sentence[1:]
            sentence.append(next_word)
            sys.stdout.write(next_word + " ")
            sys.stdout.flush()

        print(
            "\n-----------------------------------------------------------------------------------------------\n"
        )
