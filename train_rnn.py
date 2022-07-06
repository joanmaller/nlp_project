import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

import sys


#Default file 
path_to_file = 'amy-winehouse.txt'


if len(sys.argv) > 1 :
  path_to_file = sys.argv[1]
 
#Name of file  
name = path_to_file.split('.')[0]
#print(name)
data= open(path_to_file).read()
corpus = data.lower().split('\n')
print( "The first 100 words of the corpus: ", '\n', corpus[:100])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1 
index_word = {index:word for word, index in tokenizer.word_index.items()}
print("Number of words: ", total_words)
print(type(index_word))

n_gram_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequences.append(token_list[:i+1])

print(np.shape(n_gram_sequences))

#for seq in n_gram_sequences:
#  print ([index_word[w] for w in seq])

#pad sequences
max_len = max([len(seq) for seq in n_gram_sequences])
#print(max_len, total_words)

#If you use post-pad, the model kind of “forgets” what is in the beginning, so in this case, pre-pad is more powerful.
n_gram_sequences = np.array(pad_sequences(n_gram_sequences, padding='pre', maxlen=max_len))

print
#select data and labels
xs, labels = n_gram_sequences[:,:-1], n_gram_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


#Define RNN Model
num_dim = 90
model = Sequential()
model.add(Embedding(total_words, num_dim, input_length=max_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(96)))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(xs, ys, epochs=100, verbose=1)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'acc')
plot_graphs(history, 'loss')

#We train it for 50 epochs more
history2 = model.fit(xs, ys, epochs=50, verbose=1)
plot_graphs(history2, 'acc')
plot_graphs(history2, 'loss')

#Save the model
name = path_to_file.split('.')[0]
model.save(name + '.h5')








