from abc import ABC, abstractmethod
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, ConvLSTM2D, Input, BatchNormalization
from keras import optimizers
from keras.layers.merging import concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D
from enum import Enum


class IModel(ABC):

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def fit_data_on_model(self, X_train, X_test):
        pass


class ModelInfo():

    def __init__(self):

        self._model_names = [ModelName.lstm,
                             ModelName.cnn_lstm,
                             ModelName.conv_lstm,
                             ModelName.multi_head_cnn,
                             ModelName.cnn]

        self._model_names = [ModelName.lstm,
                             ModelName.cnn_lstm,
                             ModelName.conv_lstm]

        # self._model_names = [ModelName.conv_lstm]

    @property
    def model_names(self):
        return self._model_names


class ModelName(Enum):

    lstm = "LSTM"
    cnn_lstm = "CNN-LSTM"
    conv_lstm = "ConvLSTM"
    multi_head_cnn = "Multi-head-CNN"
    cnn = "CNN"


class BasicCNN(IModel):

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.hyper_model = hyper_model
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        adam = optimizers.Adam(learning_rate=self.hyper_model['learning_rate'])

        model = Sequential()
        model.add(Conv1D(filters=self.hyper_model['filters'],
                         kernel_size=self.hyper_model['kernel'],
                         activation=self.hyper_model["activation"],
                         input_shape=(self.n_timesteps, self.n_features)))
        model.add(Conv1D(filters=self.hyper_model['filters'],
                         kernel_size=self.hyper_model['kernel'],
                         activation=self.hyper_model["activation"]))
        model.add(Dropout(.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation=self.hyper_model["activation"]))
        model.add(Dense(self.n_outputs, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=adam)

        return model

    def fit_data_on_model(self, X_train, X_test):
        return X_train, X_test


class MultiHeadCNN(IModel):

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.hyper_model = hyper_model
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        adam = optimizers.Adam(learning_rate=self.hyper_model['learning_rate'])

        input_layer = Input(shape=(self.n_timesteps, self.n_features))

        # head 1
        conv1 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=3,
                       activation=self.hyper_model["activation"])(input_layer)
        conv1 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=3,
                       activation=self.hyper_model["activation"])(conv1)
        drop1 = Dropout(.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        # head 2
        conv2 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=6,
                       activation=self.hyper_model["activation"])(input_layer)
        conv2 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=6,
                       activation=self.hyper_model["activation"])(conv2)
        drop2 = Dropout(.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        # head 3
        conv3 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=12,
                       activation=self.hyper_model["activation"])(input_layer)
        conv3 = Conv1D(filters=self.hyper_model['filters'],
                       kernel_size=12,
                       activation=self.hyper_model["activation"])(conv3)
        drop3 = Dropout(.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        # merge
        merged = concatenate([flat1, flat2, flat3])

        # interpretation
        dense1 = Dense(100, activation=self.hyper_model["activation"])(merged)
        outputs = Dense(self.n_outputs, activation="softmax")(dense1)
        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer=adam)

        return model

    def fit_data_on_model(self, X_train, X_test):
        return X_train, X_test


class BasicLSTM(IModel):

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.hyper_model = hyper_model
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        adam = optimizers.Adam(learning_rate=self.hyper_model['learning_rate'])

        model = Sequential()
        model.add(LSTM(self.hyper_model["nodes"], input_shape=(self.n_timesteps, self.n_features)))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(128, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(64, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(self.n_outputs, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=adam)
        # model.summary()

        return model

    def fit_data_on_model(self, X_train, X_test):
        return X_train, X_test


class CnnLSTM(IModel):

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.hyper_model = hyper_model
        self.n_splits = 4
        self.n_length = int(n_timesteps / self.n_splits)

        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        adam = optimizers.Adam(learning_rate=self.hyper_model['learning_rate'])

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=int(self.hyper_model['filters']), kernel_size=int(self.hyper_model['kernel']), activation=self.hyper_model["activation"]),
                                  input_shape=(None, self.n_length, self.n_features)))
        model.add(TimeDistributed(Dropout(self.hyper_model["dropout_rate"])))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(self.hyper_model["nodes"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(128, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(64, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(self.n_outputs, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=adam)
        # model.summary()

        return model

    def fit_data_on_model(self, X_train, X_test):

        X_train = X_train.reshape((X_train.shape[0], self.n_splits, self.n_length, self.n_features))
        X_test = X_test.reshape((X_test.shape[0], self.n_splits, self.n_length, self.n_features))

        return X_train, X_test


class ConvLSTM(IModel):

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.hyper_model = hyper_model
        self.n_splits = 4
        self.n_length = int(n_timesteps / self.n_splits)

        self.n_features = n_features
        self.n_outputs = n_outputs

    def create(self):

        adam = optimizers.Adam(learning_rate=self.hyper_model['learning_rate'])

        model = Sequential()
        model.add(ConvLSTM2D(filters=self.hyper_model['filters'], kernel_size=(1, self.hyper_model['kernel']), activation=self.hyper_model["activation"],
                             input_shape=(self.n_splits, 1, self.n_length, self.n_features)))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))
        model.add(Flatten())

        model.add(Dense(128, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(64, activation=self.hyper_model["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(self.hyper_model["dropout_rate"]))

        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer=adam)
        # model.summary()

        return model

    def fit_data_on_model(self, X_train, X_test):

        X_train = X_train.reshape((X_train.shape[0], self.n_splits, 1, self.n_length, self.n_features))
        X_test = X_test.reshape((X_test.shape[0], self.n_splits, 1, self.n_length, self.n_features))

        return X_train, X_test


