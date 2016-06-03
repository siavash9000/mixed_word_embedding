from sklearn.base import TransformerMixin, BaseEstimator
from keras.preprocessing import sequence
from collections import Counter
from keras.models import Model
from keras.layers import Dense, Embedding, Input, TimeDistributed, LSTM,  \
    Activation
import numpy as np
from keras.datasets import reuters
from keras.utils import np_utils

UNKNOWN_ITEM_ID = 0
UNKNOWN_ITEM = u'\uFFFD'


def count_chars(texts, preprocess=None):
    char_counts = Counter()
    if preprocess is None:
        for text in texts:
            for c in text:
                char_counts[c] += 1
    else:
        for text in texts:
            text = preprocess(text)
            for c in text:
                char_counts[c] += 1
    return char_counts


def build_dictionaries(char_counts):
    # map chars to ids. we reserve the id=0 for the unknown character
    char2id = dict((c, i + 1) for i, (c, count) in enumerate(char_counts))
    char2id[UNKNOWN_ITEM] = UNKNOWN_ITEM_ID
    id2char = {i: c for c, i in char2id.iteritems()}
    return char2id, id2char


class DataAccess(object):
    def __init__(self, vocab_size=None, max_word_length=None,
                 max_text_length=None):
        self.tokenizer = Tokenizer(vocab_size=vocab_size)
        self.padder = SequencePadder(max_word_len=max_word_length,
                                 max_text_len=max_text_length)
        self.max_text_length = max_text_length

    def tokenize(self, X_test, X_train):
        vocab_char_size = self.tokenizer.fit(X_train + X_test)
        X_train = self.tokenizer.transform(X_train)
        X_test = self.tokenizer.transform(X_test)
        return X_test, X_train, vocab_char_size

    def pad(self, X_test, X_train):
        self.padder.fit(np.concatenate((X_train, X_test)))
        X_train = self.padder.pad_words(X_train)
        X_test = self.padder.pad_words(X_test)
        X_train = self.padder.pad_texts(X_train)
        X_test = self.padder.pad_texts(X_test)
        return X_test, X_train

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = reuters.load_data(
            start_char=None, index_from=None, nb_words=self.max_text_length)
        index_word = dict((v, k) for k, v in reuters.get_word_index().items())
        X_train = [[index_word[idx] for idx in x] for x in X_train]
        X_test = [[index_word[idx] for idx in x] for x in X_test]
        X_test, X_train, vocab_char_size = self.tokenize(X_test, X_train)
        X_test, X_train = self.pad(X_test, X_train)
        nb_classes = np.max(y_train)+1
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        return X_train, Y_train, X_test, Y_test, vocab_char_size, nb_classes


class CharIndexer(object):
    def __init__(self, vocab_size=None, preprocess=None):
        self.vocab_size_ = vocab_size
        self._preprocess = preprocess

    def fit_on_texts(self, texts):
        texts = ' '.join([' '.join(x) for x in texts])
        char_counts = count_chars(texts, self._preprocess)
        top_chars = char_counts.most_common()
        self._char2id, self._id2char = build_dictionaries(top_chars)

    def vocab_size(self):
       return len(self._char2id)

    def transform_word_to_sequence(self, word):
        return [self._char2id.get(c, UNKNOWN_ITEM_ID) for c in word]


class SequencePadder(BaseEstimator, TransformerMixin):

    def __init__(self, max_word_len=None, max_text_len=None):
        self.max_word_len = max_word_len
        self.max_text_len = max_text_len

    def fit(self, X, y=None):
        return self

    def pad_words(self, X, y=None):
        maxlen = self.max_word_len
        if maxlen is None:
            maxlen = max([len(w) for w in [x for x in X]])
        padded = []
        for x in X:
            padded.append(sequence.pad_sequences(x, maxlen=maxlen, value=-1))
        return np.array(padded)

    def pad_texts(self, X, y=None):
        maxlen = self.max_text_len
        if maxlen is None:
            maxlen = max([len(x) for x in X])
        return sequence.pad_sequences(X, maxlen=maxlen, value=-1)


class Tokenizer(BaseEstimator, TransformerMixin):

    def __init__(self, vocab_size=None):
        self.indexer_ = None
        self.vocab_size_char = 0

    def fit(self, X, y=None):
        self.indexer_ = CharIndexer(vocab_size=None)
        self.indexer_.fit_on_texts(X)
        print(self.summary())
        return len(self.vocab_char())

    def transform(self, X, y=None):
        tokens_character_level = []
        for x in X:
            x_char_encoded = map(lambda w:
                                 self.indexer_.transform_word_to_sequence(w),
                                 x)
            tokens_character_level.append(x_char_encoded)
        return np.array(tokens_character_level)

    def vocab_char(self):
        char_items = [item for item in self.indexer_._char2id.iteritems()]
        char_items.sort(key=lambda (c, i): i)
        chars = []
        for (k, i) in char_items:
            if isinstance(k, unicode):
                chars.append(k.encode('utf-8'))
            else:
                chars.append(k)
        return chars

    def summary(self):
        chars = self.vocab_char()
        summary = 'Character vocabulary: {}, vocabulary size: {}'\
            .format(''.join(chars), str(len(chars)))
        return summary


def build_model(nb_classes, chars_vocab_size, word_count, word_length):
    print('Build model...')
    input = Input(shape=(word_count, word_length,),dtype='int32',
                  name='input')
    embedded = TimeDistributed(Embedding(chars_vocab_size, 128,
                                         input_length=word_count,
                                         name='embedding'),
                               name='td_embedding')(input)
    forward_lstm = TimeDistributed(LSTM(64,name='char_lstm'),
                                   name='td_char_lstm')(embedded)
    # backward_lstm = LSTM(64, go_backwards=True,
    #                      sequence_of_sequences=True)(input)
    # char_embedding = merge([forward_lstm,backward_lstm],mode='concat')
    lstm = LSTM(64, name='word_lstm')(forward_lstm)
    dense = Dense(nb_classes, activation='sigmoid', name='dense')(lstm)
    output = Activation('softmax', name='output')(dense)
    model = Model(input=input, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_and_test_model(X_train, y_train, X_test, y_test, batch_size, nb_epoch = 15):
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

WORD_VOCAB_SIZE = 20000
WORD_COUNT = 1000
WORD_LENGTH = 20
BATCH_SIZE = 32
CHAR_VOCAB_SIZE = 40

data_access = DataAccess(vocab_size=WORD_VOCAB_SIZE, max_word_length=WORD_LENGTH,
                  max_text_length=WORD_COUNT, )

X_train, y_train, X_test, y_test, vocab_char_size, nb_classes = data_access.load_data()

model = build_model(nb_classes=nb_classes,
                    chars_vocab_size=CHAR_VOCAB_SIZE,
                    word_count=WORD_COUNT,
                    word_length=WORD_LENGTH)

train_and_test_model(X_train, y_train, X_test, y_test, BATCH_SIZE)