###########################################################
# SMDL Final Project Template (100 marks total)
###########################################################
# 1. Please complete all sections
# 2. Minimum requirements: implement and train a model using Tensorflow (90 marks)
#    Example project ideas:
#    - Text Prediction or Generation
#    - Time Series Prediction or Generation
#    - Video Classification
#    - Neural Machine Translation
# 3. Bonus task: call your model from a Flask application (10 marks)
#
###########################################################
# Section 1 (10 marks)
# Name (matching course registration):
# Matthew Hou

# Title of Project:
# Amazon review auto-complete sentence
#
# Business Problem to be Solved:
# When customers leave reviews/rate Amazon products, this model aims to assist them to complete their reviews
#
# Dataset Source(s):
#https://www.kaggle.com/bittlingmayer/amazonreviews/notebooks
#
# Model Inputs:
#  Existing text in the (un-completed) sentence
#
# Model Targets:
# Predicts the next word in the sentence. This is looped for multiple predictions
#
###########################################################
# Section 2 (20 marks)
# Preprocessing (python code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, GRU, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import os
import pickle
import numpy as np
import pandas as pd
import bz2
import gc
import chardet
import re
import os
from random import sample
import time

t1=time.time()

MODEL_ARTIFACTS = dict()
MODEL_DIR = os.path.join('..', 'app', 'demo', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

def save_artifacts(key_values: dict, dest='model_artifacts_negative.pkl'):
    MODEL_ARTIFACTS.update(key_values)
    pickle.dump(MODEL_ARTIFACTS, open(os.path.join(MODEL_DIR, dest), 'wb'))


# we'll just use the training set for this task because we are predicting
# the next word (generating our own target).
train_file = bz2.BZ2File('train.ft.txt.bz2')

def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]))
        texts.append(x[10:].strip())
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')

negative_text = [i for (i, v) in zip(train_texts, train_labels) if v==1]

#Run this again for positive_text model
#positive_text = [i for (i, v) in zip(train_texts, train_labels) if v==2]

negative_text=sample(negative_text,20000)

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, lower=True)
tokenizer.fit_on_texts(negative_text)
sequences = tokenizer.texts_to_sequences(negative_text)

model_text = [i for i in sequences if len(i)<=80]

#sequence_len = int(np.median(np.array([len(s) for s in model_text])))
sequence_len=48 #Set 'hardcopy' to ensure number remains the same when re-running

save_artifacts({'sequence_len': sequence_len})


padded_len = int(sequence_len*1.2)
sequences = pad_sequences(model_text, maxlen=padded_len, padding='pre')

X = []
y = []
for s in sequences:
    for j in range(len(s) - sequence_len):
        X.append(np.array(s[j:j+sequence_len]))
        y.append(s[j+sequence_len])

X = np.array(X)
y_cat = np.array(to_categorical(y))

print(X.shape,y_cat.shape)
print(sequence_len)

tokenizer_config = json.loads(tokenizer.to_json())
save_artifacts({'tokenizer_config': tokenizer_config})

###########################################################
# Section 3 (40 marks)
# Model Architecture and Training (python code)
 
embedding_len = 100
batch_size = 64
num_outputs = y_cat.shape[1]

model_file = os.path.join(MODEL_DIR, 'gru_negative.h5')
if os.path.isfile(model_file):
    # load model if it already exists, so that we can continue training
    model = load_model(model_file)

else:
    model_input = Input(shape=(sequence_len,), dtype='int64')
    x = Embedding(vocab_size, embedding_len, input_length=sequence_len)(model_input)
    x = GRU(16,activation='tanh',return_sequences=True)(x)
    x = GRU(32, activation='tanh', return_sequences=True)(x)
    x = GRU(64, activation='tanh', return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(num_outputs,activation='softmax')(x)
    model = Model(model_input, x)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

mc = ModelCheckpoint(model_file,
                     monitor='acc', save_best_only=True)

history = model.fit(X, y_cat, epochs=12, batch_size=batch_size
                    ,callbacks=[mc])

plt.plot(history.history['acc'], label='train')
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('learning_curve.png')
plt.show()


best_model = load_model(model_file)

start = 'This item is really lousy'

test_seqs = tokenizer.texts_to_sequences([start])
for i in range(10):
    test_seqs_padded = pad_sequences(test_seqs, maxlen=sequence_len,
                                     padding='pre', truncating='pre')

    next_word = best_model.predict(test_seqs_padded)
    next_word=next_word.argmax(axis=1)
    test_seqs[0].append(next_word[0])
    print(tokenizer.sequences_to_texts(test_seqs))


print(time.time()-t1)


###########################################################
# Section 4 (10 marks)
# Observations (written)
# I averaged ~15% accuracy only, this is evident from relatively poor text prediction
# Key limitation seems to be training time. I had to limit X_train size
# This took around 20min
#
###########################################################
# Section 5 (10 marks)
# Further Improvements (written)
# Additional training time would be good
# I would also  suggest choosing more specific data (e.g. different models for different product categories)
#
#
