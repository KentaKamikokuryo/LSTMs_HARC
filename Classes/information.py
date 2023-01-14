import os


class PathInfo:

    def __init__(self):

        self.cwd = os.getcwd()
        self.cwd = "C:\\Users\\Kenta Kamikokuryo\\Desktop\\Research\\GIT_Projects\\LSTMs_HARC"
        print("cwd: {0}".format(self.cwd))

        self._set_data_path()

        self._set_save_folder()

    def _set_data_path(self):

        self._path_data = self.cwd + "\\HARDataset\\"
        print("path for data: {}".format(self._path_data))

        self._file_path_test, self._filenames_test = Utilities.load_filenames(group="test",
                                                                              prefix=self._path_data)
        print("path for file test: {0}".format(self._file_path_test))

        self._file_path_train, self._filenames_train = Utilities.load_filenames(group="train",
                                                                                prefix=self._path_data)
        print("path for file train: {0}".format(self._file_path_train))

    def _set_save_folder(self):

        self._folder_ML = self.cwd + "\\ML\\"
        if not (os.path.exists(self._folder_ML)):
            os.makedirs(self._folder_ML)

        self._path_figure = self._folder_ML + "Figures\\"
        if not (os.path.exists(self._path_figure)):
            os.makedirs(self._path_figure)

        self._path_search = self._folder_ML + "Search_Models\\"
        if not (os.path.exists(self._path_search)):
            os.makedirs(self._path_search)

        self._path_results = self._folder_ML + "Results\\"
        if not (os.path.exists(self._path_results)):
            os.makedirs(self._path_results)

    def set_path_figure_model(self, model_name):

        path_figure_model = self._path_figure + model_name + "\\"
        if not (os.path.exists(path_figure_model)):
            os.makedirs(path_figure_model)

        return path_figure_model

    def set_path_search_model(self, model_name):

        path_search_model = self._path_search + model_name + "\\"
        if not (os.path.exists(path_search_model)):
            os.makedirs(path_search_model)

        return path_search_model

    @property
    def path_data(self):
        return self._path_data

    @property
    def file_path_test(self):
        return self._file_path_test

    @property
    def file_path_train(self):
        return self._file_path_train

    @property
    def filenames_test(self):
        return self._filenames_test

    @property
    def filenames_train(self):
        return self._filenames_train

    @property
    def path_results(self):
        return self._path_results


class Utilities:

    @staticmethod
    def load_filenames(group, prefix=""):

        FILE_FORMAT = ".txt"
        FILE_TYPEs = ["total_acc", "body_acc", "body_gyro"]
        DIRECTIONs = ["_x_", "_y_", "_z_"]

        filepath = prefix + group + "\\Inertial Signals\\"

        # load all 9 files as a single array
        filenames = list()

        # total acceleration
        filenames += [FILE_TYPEs[0] + DIRECTIONs[0] + group + FILE_FORMAT,
                      FILE_TYPEs[0] + DIRECTIONs[1] + group + FILE_FORMAT,
                      FILE_TYPEs[0] + DIRECTIONs[2] + group + FILE_FORMAT]

        filenames += [FILE_TYPEs[1] + DIRECTIONs[0] + group + FILE_FORMAT,
                      FILE_TYPEs[1] + DIRECTIONs[1] + group + FILE_FORMAT,
                      FILE_TYPEs[1] + DIRECTIONs[2] + group + FILE_FORMAT]

        filenames += [FILE_TYPEs[2] + DIRECTIONs[0] + group + FILE_FORMAT,
                      FILE_TYPEs[2] + DIRECTIONs[1] + group + FILE_FORMAT,
                      FILE_TYPEs[2] + DIRECTIONs[2] + group + FILE_FORMAT]

        return filepath, filenames


