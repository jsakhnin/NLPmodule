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

# This currently uses only the glove word embedding file
EMBEDDING_FILES = [
    '../glove.840B.300d.gensim'
]
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
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

def build_matrix(word_index, path):
    embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]:
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

print("Loading data")
df = pd.read_csv('https://media.githubusercontent.com/media/jsakhnin/JigsawNLP_data/master/train.csv') #, nrows=1000)
print("Loading data complete")

#for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
#    df[column] = np.where(df[column] >= 0.5, True, False)

print("Set up training and test data")
#comments   = df[TEXT_COLUMN].astype(str)
#labels     = df[TARGET_COLUMN].values
#aux_labels = df[AUX_COLUMNS].values

#x_train, x_test, y_train, y_test, y_aux_train, y_aux_test = train_test_split(comments, labels, aux_labels,
#                                                                             train_size=0.8, random_state=seed,
#                                                                             shuffle=True)

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

print("Setting up initial weights for model")
sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train[TARGET_COLUMN] * (~train[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train[TARGET_COLUMN]) * train[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights /= sample_weights.mean()

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
print("Finished setting up weights")

print("Training the models")

LOGNAME = "model-{}Epochs-{}".format(EPOCHS, int(time.time()))
#tensorboard = TensorBoard(log_dir='logs\{}'.format(LOGNAME))
tensorboard = TensorBoard(log_dir='logs')

model = build_model(embedding_matrix, y_aux_train.shape[-1])
model.fit(
    x_train,
    [y_train, y_aux_train],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=2,
    sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
    callbacks=[tensorboard]
)
model.save(f"simple_lstm_model_{EPOCHS}Epochs.h5")

print("Finished training the models")

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

submission = pd.DataFrame.from_dict({
    'y_test': y_test,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)

#model2 = load_model(f"simple_lstm_model_{EPOCHS}Epochs.h5")
#predictions = model2.predict(x_test, batch_size=2048)[0].flatten()
#predictions = np.where(predictions >= 0.5, True, False)
#acc = accuracy_score(y_test, predictions)
#print(f"Classification Accuracy on loaded model = {acc}")
