from ClassesML.models import Model, BasicLSTM, CnnLSTM, ConvLSTM
import sys


class Factory():

    def __init__(self, model_name, n_timesteps, n_features, n_outputs):

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model_name = model_name

    def create(self):

        if self.model_name == Model.lstm:

            model = BasicLSTM(n_timesteps=self.n_timesteps,
                              n_features=self.n_features,
                              n_outputs=self.n_outputs)

        elif self.model_name == Model.cnn_lstm:

            model = CnnLSTM(n_timesteps=self.n_timesteps,
                            n_features=self.n_features,
                            n_outputs=self.n_outputs)

        elif self.model_name == Model.conv_lstm:

            model = ConvLSTM(n_timesteps=self.n_timesteps,
                             n_features=self.n_features,
                             n_outputs=self.n_outputs)

        else:

            sys.exit(1)

        return model

