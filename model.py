import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

TRAIN_CSV = 'data/preprocessed_train.csv'
TEST_CSV = 'data/preprocessed_test.csv'
MODEL_PATH = 'models/sentiment_lstm.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

MAX_VOCAB = 10000
MAX_SEQUENCE_LEN = 100
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

def train_lstm_model():
    train_data = pd.read_csv(TRAIN_CSV)
    test_data = pd.read_csv(TEST_CSV)

    train_data = train_data.dropna(subset=['text', 'sentiment'])
    test_data = test_data.dropna(subset=['text', 'sentiment'])

    input_texts_train = train_data['text'].astype(str).values
    sentiment_labels_train = train_data['sentiment'].astype(str).values

    input_texts_test = test_data['text'].astype(str).values
    sentiment_labels_test = test_data['sentiment'].astype(str).values

    label_encoder = LabelEncoder()
    encoded_labels_train = label_encoder.fit_transform(sentiment_labels_train)
    encoded_labels_test = label_encoder.transform(sentiment_labels_test)

    onehot_labels_train = to_categorical(encoded_labels_train)
    onehot_labels_test = to_categorical(encoded_labels_test)

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(input_texts_train)

    train_sequences = tokenizer.texts_to_sequences(input_texts_train)
    test_sequences = tokenizer.texts_to_sequences(input_texts_test)

    padded_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LEN, padding='post')
    padded_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN, padding='post')

    model = Sequential([
        Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN),
        LSTM(128, return_sequences=False, implementation=1),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(onehot_labels_train.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        padded_train, onehot_labels_train,
        validation_data=(padded_test, onehot_labels_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, 'wb') as tfile:
        pickle.dump(tokenizer, tfile)

    with open(ENCODER_PATH, 'wb') as efile:
        pickle.dump(label_encoder, efile)

def predict_sentiment(text_input):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, 'rb') as tfile:
        tokenizer = pickle.load(tfile)

    with open(ENCODER_PATH, 'rb') as efile:
        label_encoder = pickle.load(efile)

    sequence = tokenizer.texts_to_sequences([text_input])
    padded_input = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding='post')

    prediction_probs = model.predict(padded_input)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label

train_lstm_model()
