import rnn_utils
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Embedding
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras import regularizers
from keras import metrics
import keras.callbacks as cb

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def init_model():
    model = Sequential()
    model.add(LSTM(178, activation='relu', input_shape = (23,178), return_sequences=True))
    model.add(LSTM(178, activation='relu', return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])
    return model

def plot_losses(losses):
    plt.plot(losses)
    plt.title('Loss per batch')
    plt.show()

# the training set is stored in this directory
filePath = 'C:/Users/zheng/Dropbox/ima camp/data science/project/' 
#filePath = '/Users/zhengao/Dropbox/ima camp/data science/project'
EEGData= rnn_utils.loadData(filePath)
EEGData_preprocessed = rnn_utils.preprocessData(EEGData)
(X_train, y_train), (X_test, y_test) = EEGData_preprocessed

#print(X_train.shape)
#print(y_test.shape)

(X_train_mini_batch, y_train_mini_batch) = rnn_utils.mini_batch(X_train, y_train, 0, 0)
(X_test_mini_batch, y_test_mini_batch) = rnn_utils.mini_batch(X_test, y_test, 0, 0)

#print(X_train_mini_batch.shape)
#print(y_test_mini_batch.shape)

model = init_model()
history = LossHistory()
model.fit(X_train_mini_batch, y_train_mini_batch, epochs=100, batch_size=512,
                callbacks=[history],
                validation_data=(X_test_mini_batch, y_test_mini_batch), verbose=1)
score = model.evaluate(X_test_mini_batch, y_test_mini_batch, batch_size=16)
print(score)
prediction = model.predict(X_test_mini_batch)
plot_losses(history.losses)

