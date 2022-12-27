from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv1D, MaxPooling1D


class IModel(ABC):

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def fit_data_on_model(self, X_train, X_test):
        pass


class ModelInfo:

    def __init__(self):

        self._model_names = [Model.lstm,
                             Model.cnn_lstm,
                             Model.conv_lstm]

    @property
    def model_names(self):
        return self._model_names


class Model:

    lstm = "LSTM"
    cnn_lstm = "CNN LSTM"
    conv_lstm = "ConvLSTM"


class BasicLSTM(IModel):

    def __init__(self, n_timesteps, n_features, n_outputs):

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        model = Sequential()
        model.add(BasicLSTM(100, input_shape=(self.n_timesteps, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(self.n_outputs, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def fit_data_on_model(self, X_train, X_test):
        return X_train, X_test


class CnnLSTM(IModel):

    def __init__(self, n_timesteps, n_features, n_outputs):

        self.n_splits = 4
        self.n_length = int(n_timesteps / self.n_splits)

        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation="relu"),
                                  input_shape=(None, self.n_length, self.n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation="relu")))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(self.n_outputs, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def fit_data_on_model(self, X_train, X_test):

        X_train = X_train.reshape((X_train.shape[0], self.n_splits, self.n_length, self.n_features))
        X_test = X_test.reshape((X_test.shape[0], self.n_splits, self.n_length, self.n_features))

        return X_train, X_test


class ConvLSTM(IModel):

    def __init__(self, n_timesteps, n_features, n_outputs):

        self.n_splits = 4
        self.n_length = int(n_timesteps / self.n_splits)

        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                             input_shape=(self.n_splits, 1, self.n_length, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def fit_data_on_model(self, X_train, X_test):

        X_train = X_train.reshape((X_train.shape[0], self.n_splits, 1, self.n_length, self.n_features))
        X_test = X_test.reshape((X_test.shape[0], self.n_splits, 1, self.n_length, self.n_features))

        return X_train, X_test


