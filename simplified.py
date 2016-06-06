from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dense, Embedding, Input, TimeDistributed, LSTM,  \
    Activation
import numpy as np


def main():
    DATA_SIZE = 1000
    BATCH_SIZE = 100
    WORD_COUNT = 20
    WORD_LENGTH = 10
    CHAR_VOCAB_SIZE = 50
    NB_CLASSES = 5

    X = np.random.randint(CHAR_VOCAB_SIZE, size=(DATA_SIZE, WORD_COUNT, WORD_LENGTH))
    Y = np.random.randint(NB_CLASSES, size=DATA_SIZE)
    Y = np_utils.to_categorical(Y, NB_CLASSES)

    input = Input(batch_shape=(BATCH_SIZE,WORD_COUNT, WORD_LENGTH, ),
                  dtype='int32')
    embedded = TimeDistributed(Embedding(CHAR_VOCAB_SIZE, 128,
                                         input_length=WORD_COUNT))(input)
    char_lstm = TimeDistributed(LSTM(64, consume_less='cpu'))(embedded)
    lstm = LSTM(64,consume_less='cpu')(char_lstm)
    dense = Dense(NB_CLASSES, activation='sigmoid')(lstm)
    output = Activation('softmax')(dense)
    model = Model(input=input, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=5)


if __name__ == "__main__":
    main()