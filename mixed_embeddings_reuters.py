from sklearn.base import TransformerMixin, BaseEstimator
from keras.preprocessing import sequence
from collections import Counter
from keras.models import Model
from keras.layers import Dense, Embedding, Input, TimeDistributed, GRU,  \
    Activation, merge
import numpy as np
from keras.datasets import reuters
from keras.utils import np_utils
from operator import itemgetter
import random

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
    def __init__(self, max_word_length=None, word_vocab_size=None,
                 max_text_length=None):
        self.tokenizer = Tokenizer()
        self.padder = SequencePadder(max_word_len=max_word_length,
                                     max_text_len=max_text_length)
        self.word_vocab_size = word_vocab_size

    def tokenize(self, X_test, X_train):
        vocab_char_size = self.tokenizer.fit(X_train + X_test)
        X_train = self.tokenizer.transform(X_train)
        X_test = self.tokenizer.transform(X_test)
        return X_test, X_train, vocab_char_size

    def pad(self,X_test_char, X_train_char,X_test, X_train):
        self.padder.fit(np.concatenate((X_train, X_test)))
        X_train_char = self.padder.pad_words(X_train_char)
        X_test_char = self.padder.pad_words(X_test_char)
        X_train_char = self.padder.pad_texts(X_train_char)
        X_test_char = self.padder.pad_texts(X_test_char)

        X_train = self.padder.pad_texts(X_train)
        X_test = self.padder.pad_texts(X_test)
        return X_test, X_train, X_test_char, X_train_char

    def load_data(self, sample_size=None):
        print('Load Data...')
        (X_train, y_train), (X_test, y_test) = reuters.load_data(
            start_char=None, index_from=None, nb_words=self.word_vocab_size)
        if sample_size:
            sample_indices_train = random.sample(range(len(X_train)),
                                                 sample_size)
            X_train = itemgetter(*sample_indices_train)(X_train)
            y_train = itemgetter(*sample_indices_train)(y_train)

            sample_indices_test = random.sample(range(len(X_test)),sample_size)
            X_test = itemgetter(*sample_indices_test)(X_test)
            y_test = itemgetter(*sample_indices_test)(y_test)
        index_word = dict((v, k) for k, v in reuters.get_word_index().items())
        X_train_char = [[index_word[idx] for idx in x] for x in X_train]
        X_test_char = [[index_word[idx] for idx in x] for x in X_test]
        X_test_char, X_train_char, vocab_char_size = \
            self.tokenize(X_test_char, X_train_char)
        X_test, X_train, X_test_char, X_train_char = \
            self.pad(X_test_char, X_train_char,X_test, X_train)
        nb_classes = np.max(y_train+y_test)+1
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        return X_train, X_train_char, Y_train, X_test, X_test_char, Y_test, \
               vocab_char_size, nb_classes


class CharIndexer(object):
    def __init__(self, preprocess=None):
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
        padded = []
        for x in X:
            padded.append(sequence.pad_sequences(x, maxlen=self.max_word_len, value=-1))
        return np.array(padded)

    def pad_texts(self, X, y=None):
        return sequence.pad_sequences(X, maxlen=self.max_text_len, value=-1)


class Tokenizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.indexer_ = None

    def fit(self, X, y=None):
        self.indexer_ = CharIndexer()
        self.indexer_.fit_on_texts(X)
        return self.indexer_.vocab_size()

    def transform(self, X, y=None):
        tokens_character_level = []
        for x in X:
            x_char_encoded = map(lambda w:
                                 self.indexer_.transform_word_to_sequence(w),
                                 x)
            tokens_character_level.append(x_char_encoded)
        return np.array(tokens_character_level)


def build_model(nb_classes, word_vocab_size, chars_vocab_size,
                word_count, word_length, batch_size):
    print('Build model...')
    CONSUME_LESS='gpu'
    char_input = Input(batch_shape=(batch_size,word_count, word_length,),
            dtype='int32', name='char_input')
    character_embedding = TimeDistributed(Embedding(chars_vocab_size, 15,
                                         input_length=word_length,
                                         name='char_embedding'),
                                          name='td_char_embedding')(char_input)
    forward_gru = TimeDistributed(GRU(16,name='char_gru_forward',
                                        consume_less=CONSUME_LESS),
                                   name='td_char_gru_forward')(character_embedding)
    backward_gru = TimeDistributed(GRU(16,name='char_gru_backward',
                                        consume_less=CONSUME_LESS,
                                        go_backwards=True),
                                   name='td_char_gru_backward')(character_embedding)
    char_embedding = merge([forward_gru,backward_gru],mode='concat')

    word_input = Input(batch_shape=(batch_size,word_count,),
            dtype='int32', name='word_input')
    word_embedding = Embedding(word_vocab_size, 32,
                                         input_length=word_count,
                                         name='word_embedding')(word_input)

    embedding = merge([char_embedding,word_embedding],mode='concat')
    word_gru = GRU(32, name='word_lstm', consume_less=CONSUME_LESS)(embedding)
    dense = Dense(nb_classes, activation='sigmoid', name='dense')(word_gru)
    output = Activation('softmax', name='output')(dense)
    model = Model(input=[char_input,word_input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_and_test_model(X_train, X_train_char, Y_train, X_test,
                         X_test_char, Y_test, batch_size, nb_epoch = 200):
    print('Train and Test model...')
    model.fit({'char_input': X_train_char, 'word_input': X_train},
              {'output': Y_train},
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=([X_test_char,X_test], Y_test))
    score, acc = model.evaluate([X_test_char,X_test], Y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

WORD_VOCAB_SIZE = 20000
MAX_TEXT_LENGTH = 80
MAX_WORD_LENGTH = 12
BATCH_SIZE = 256

data_access = DataAccess(max_word_length=MAX_WORD_LENGTH,
                         word_vocab_size=WORD_VOCAB_SIZE,
                         max_text_length=MAX_TEXT_LENGTH)

X_train, X_train_char, Y_train, X_test, X_test_char, Y_test, \
            vocab_char_size, nb_classes = data_access.load_data()

model = build_model(nb_classes=nb_classes, word_vocab_size=WORD_VOCAB_SIZE,
                    chars_vocab_size=vocab_char_size,
                    word_count=MAX_TEXT_LENGTH,
                    word_length=MAX_WORD_LENGTH,
                    batch_size=BATCH_SIZE)

train_and_test_model(X_train, X_train_char, Y_train, X_test, X_test_char,
                     Y_test, BATCH_SIZE)