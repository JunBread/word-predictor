from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

text = ""

with open('news-title.txt', encoding='utf-8') as f:
    text = f.read()

t = Tokenizer(lower=False, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
t.fit_on_texts([text])
vocab_size = len(t.word_index) + 1
print('단어집합 크기: %d' % vocab_size)

sequences = list()
for line in text.split('\n'):
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

index_to_word = {}
for key, value in t.word_index.items():
    index_to_word[value] = key

print('빈도수 상위 582번 단어: {}'.format(index_to_word[582]))

max_len = max(len(l) for l in sequences)
print('샘플 최대 길이: {}'.format(max_len))

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)

if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_len - 1))

    model.add(LSTM(32))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=2)
    model.save('model.h5')

def sentence_generation(model, t, current_word):
    while True:
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen = max_len - 1, padding='pre')
        result = model.predict_classes(encoded, verbose=0)

        for word, index in t.word_index.items():
            if index == result:
                break
        if word == '<E>':
            break
        current_word += ' ' + word
    return current_word

print(sentence_generation(model, t, sys.argv[1]))