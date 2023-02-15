import pandas as pd
import itertools
from tabulate import tabulate
from ClassesML.models import ModelName


class HyperParameters:

    def __init__(self, model_name):

        self.model_name = model_name

    def generate_hyper_model(self, display=False):

        grid = {"activation": ["relu"],
                "batch_size": [32, 64],
                "dropout_rate": [.3, .5],
                'patience': [10, 15],
                "epoch": [100],
                'learning_rate': [0.005, 0.001]}

        # grid = {"activation": ["relu"],
        #         "optimizer": ["adam"],
        #         "batch_size": [16],
        #         "epoch": [2]}

        if self.model_name == ModelName.lstm:

            grid["model_name"] = [ModelName.lstm.value]
            grid["nodes"] = [64, 128]

        elif self.model_name == ModelName.cnn_lstm:

            grid["model_name"] = [ModelName.cnn_lstm.value]
            grid['kernel'] = [3, 6]
            grid['filters'] = [48, 64]
            grid["nodes"] = [64, 128]

        elif self.model_name == ModelName.conv_lstm:

            grid["model_name"] = [ModelName.conv_lstm.value]
            grid['kernel'] = [3, 6]
            grid['filters'] = [48, 64]
            grid["nodes"] = [64, 128]

        elif self.model_name == ModelName.multi_head_cnn:

            grid["model_name"] = [ModelName.multi_head_cnn.value]
            grid['filters'] = [48, 64]

        elif self.model_name == ModelName.cnn:

            grid['model_name'] = [ModelName.cnn.value]
            grid['kernel'] = [3, 6]
            grid['filters'] = [32, 48, 64]

        else:

            grid["model_name"] = "None"

        keys, values = zip(*grid.items())
        grid_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(grid_combination)

        if display:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return grid_combination, grid