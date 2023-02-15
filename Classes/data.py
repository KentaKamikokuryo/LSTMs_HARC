import pandas as pd
import numpy as np
import sys
from Classes.information import PathInfo
from keras.utils import to_categorical


class Data:

    def __init__(self, pathInfo: PathInfo):

        self.pathInfo = pathInfo

        self._X_train, self._y_train, self._subject_train = self._load_dataset_group(self.pathInfo.file_path_train,
                                                                                     self.pathInfo.filenames_train)

        self._X_test, self._y_test, self._subject_test = self._load_dataset_group(self.pathInfo.file_path_test,
                                                                                  self.pathInfo.filenames_test)

        # one hot encode y
        # self._one_hot_encode()

        self._class_names = ["Walking", "Upstairs", "Downstairs",
                             "Sitting", "Standing", "Laying"]

    def _one_hot_encode(self):

        self._y_train = to_categorical(self._y_train)
        self._y_test = to_categorical(self._y_test)

    def _load_dataset_group(self, file_path, filenames):

        loaded = list()
        for name in filenames:
            data = Data.load_file(file_path + name)
            loaded.append(data)
        X = np.dstack(loaded)

        if "test" in file_path:
            y = Data.load_file(self.pathInfo.path_data + "test\\y_test.txt")
            subjects = Data.load_file(self.pathInfo.path_data + "test\\subject_test.txt")

        elif "train" in file_path:
            y = Data.load_file(self.pathInfo.path_data + "train\\y_train.txt")
            subjects = Data.load_file(self.pathInfo.path_data + "train\\subject_train.txt")

        else:
            sys.exit(1)

        subjects = subjects.flatten()

        # zero offset class values
        y = y - 1

        return X, y, subjects

    @staticmethod
    def load_file(file_path):
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        return df.values

    @property
    def all_data(self):
        return self._X_train, self._y_train, self._X_test, self._y_test

    @property
    def class_names(self):
        return self._class_names

    @property
    def subject_train_test(self):
        return self._subject_train, self._subject_test
