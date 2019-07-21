from keras.models import Model
from keras.layers import LSTM, Input, Dropout, Dense, Embedding, Activation


def rnn_model(max_len, vocab_size):

    Inputs = Input(shape=(max_len,), dtype="float")

    Embeddings = Embedding(
        vocab_size, 512, dtype="float", input_length=max_len, trainable=True
    )
    Embeddings = Embeddings(Inputs)

    X = LSTM(256, input_shape=(max_len, 512), return_sequences=True)(Embeddings)
    X = LSTM(128, return_sequences=False)(X)
    X = Dense(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(256)(X)
    X = Dense(vocab_size)(X)
    X = Activation("softmax")(X)

    model = Model(inputs=Inputs, outputs=X)

    return model
