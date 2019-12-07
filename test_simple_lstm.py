import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
import time
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

EPOCHS = 6
MAX_LEN = 220
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
seed = 42

print("Loading data")
#df = pd.read_csv('../NLPData/train.csv') #, nrows=1000)
df = pd.read_csv('../NLPData/train_clean.csv') #, nrows=1000)
print("Loading data complete")


print("Set up training and test data")
train, test = train_test_split(df, train_size=0.8, random_state=seed, shuffle=True)

x_train     = train[TEXT_COLUMN].astype(str)
x_test      = test[TEXT_COLUMN].astype(str)
y_train     = train[TARGET_COLUMN].values
y_test      = test[TARGET_COLUMN].values
y_aux_train = train[AUX_COLUMNS].values
y_aux_test  = test[AUX_COLUMNS].values

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train[column] = np.where(train[column] >= 0.5, True, False)
    test[column]  = np.where(test[column] >= 0.5, True, False)

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
print("Finished setting up data")


print("Load model and test on test samples")
#model = load_model(f"simple_lstm_model_{EPOCHS}Epochs_raw_text.h5")
model = load_model(f"simple_lstm_model_{EPOCHS}Epochs_cleaned_data2.h5")
predictions = model.predict(x_test, batch_size=2048)[0].flatten()

predictions = np.where(predictions >= 0.5, True, False)
y_test      = np.where(y_test >= 0.5, True, False)

acc    = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
pres   = precision_score(y_test, predictions)
print(f"Classification Accuracy = {acc}")
print(f"Classification Recall = {recall}")
print(f"Classification Precision = {pres}")
print(f"Classification Report:")
print(classification_report(y_test, predictions))
print("Finished loading model and testing on test samples")


def performance(y_test, predictions):
    acc    = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    pres   = precision_score(y_test, predictions)
    print(f"Classification Accuracy = {acc}")
    print(f"Classification Recall = {recall}")
    print(f"Classification Precision = {pres}")
    print(f"Classification Report:")
    print(classification_report(y_test, predictions))

print("Start looking at performance for different demographics")
test['y_test'] = y_test
test['predictions'] = predictions

for ident in IDENTITY_COLUMNS:
    print("*****************************************************************")
    print(f"Test performance on comments label as {ident}")
    print("*****************************************************************")
    demo = test.loc[test[ident] == True]
    print(demo)
    performance(demo['y_test'].values, demo['predictions'].values)
    print()
    print()
