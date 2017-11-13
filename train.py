import numpy as np
import data_helpers
from w2v import train_word2vec
from keras.models import Sequential, Model
from keras.layers import *

embedding_dim = 150 # The size of the word vectors

batch_size = 32 # Batch Size for Neural Network training
num_epochs = 1 # Number of Epochs to run training
val_split = 0.2 # Percentage of data to be used for validation

min_word_count = 1 # Minimum word count in sentences
context_window_size = 12 # Size for context window

x, y, vocabulary, vocabulary_inv = data_helpers.load_data() # Load training data and vocabulary from data_helpers

embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context_window_size) # Train or Load weights for Word Vectors

# Shuffle the data
shuffle_indices = np.random.permutation(np.arange(len(y))) 
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: " + str(len(vocabulary)))

# Declare the model
model = Sequential()
model.add(Embedding(len(vocabulary), embedding_dim, input_length=119, weights=embedding_weights))
model.add(Conv1D(128, 3, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summarize the model
model.summary()

# Train & Save the model
model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1)
model.save("model_review.h5")
