import pandas as pd
import itertools
from tabulate import tabulate
from ClassesML.models import Model


class HyperParameters:

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hyper_model(self, display=False):

        grid = {"activation": ["tanh", "relu", "selu"],
                "optimizer": ["sgd", "adam"],
                "batch_size": [16, 32, 64],
                "epoch": [20, 40]}

        grid = {"activation": ["relu"],
                "optimizer": ["adam"],
                "batch_size": [64],
                "epoch": [10]}

        if self.model_name == Model.lstm:

            grid["model_name"] = [Model.lstm.value]

        elif self.model_name == Model.cnn_lstm:

            grid["model_name"] = [Model.cnn_lstm.value]

        elif self.model_name == Model.conv_lstm:

            grid["model_name"] = [Model.conv_lstm.value]

        else:

            grid["model_name"] = "None"

        keys, values = zip(*grid.items())
        grid_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(grid_combination)

        if display:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return grid_combination, grid