import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM
from keras.utils.data_utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras import utils

# Максимальное количество слов
num_words = 10000

# Максимальная длина новости
max_news_len = 30

# Количество классов новостей
nb_classes = 23

train = pd.read_csv('../train.csv',
                    header=[0, 1, 2], ).dropna()
news = train['text'].squeeze().tolist()
y_train = utils.to_categorical(train['class'] - 1, nb_classes)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(news)
sequences = tokenizer.texts_to_sequences(news)
x_train = pad_sequences(sequences, maxlen=max_news_len)

# print(x_train)

model_lstm = Sequential()
model_lstm.add(Embedding(num_words, 32, input_length=max_news_len))
model_lstm.add(LSTM(16))
model_lstm.add(Dense(23, activation='softmax'))
model_lstm.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# # print(model_lstm.summary())
model_lstm_save_path = 'best_model_lstm.h5'
checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

history_lstm = model_lstm.fit(x_train,
                              y_train,
                              epochs=5,
                              batch_size=128,
                              validation_split=0.1,
                              callbacks=[checkpoint_callback_lstm])

test = pd.read_csv('../test.csv',
                   header=[0, 1, 2], ).dropna()

test_sequences = tokenizer.texts_to_sequences(test['text'].squeeze().tolist())
x_test = pad_sequences(test_sequences, maxlen=max_news_len)
y_test = utils.to_categorical(test['class'] - 1, nb_classes)
model_lstm.load_weights(model_lstm_save_path)
print(model_lstm.evaluate(x_test, y_test, verbose=1))
