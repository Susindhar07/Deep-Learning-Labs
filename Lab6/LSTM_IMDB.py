import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
print(f"TensorFlow Version: {tf.__version__}")

max_features = 10000
maxlen = 256
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print("Pad sequences (samples x time)")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

embedding_dim = 128

model_lstm = Sequential([
    Embedding(max_features, embedding_dim, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.summary()
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs_lstm = 5
batch_size_lstm = 32

print("Training LSTM model...")
history_lstm = model_lstm.fit(x_train, y_train,
                              batch_size=batch_size_lstm,
                              epochs=epochs_lstm,
                              validation_data=(x_test, y_test))

print("LSTM Model Training Finished!")
print("Evaluating LSTM model...")
loss_lstm, accuracy_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)

print(f"\nTest Loss: {loss_lstm:.4f}")
print(f"Test Accuracy: {accuracy_lstm*100:.2f}%")