from ClassesML.models import Model, BasicLSTM, CnnLSTM, ConvLSTM
from ClassesML.behaviors import CodeBehavior, HyperParameterTuning, ModelsComparing, ModelRunning
import sys


class Factory():

    def __init__(self, hyper_model, n_timesteps, n_features, n_outputs):

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hyper_model = hyper_model
        self.model_name = hyper_model["model_name"]

    def create(self):

        if self.model_name == Model.lstm.value:

            model = BasicLSTM(hyper_model=self.hyper_model,
                              n_timesteps=self.n_timesteps,
                              n_features=self.n_features,
                              n_outputs=self.n_outputs)

        elif self.model_name == Model.cnn_lstm.value:

            model = CnnLSTM(hyper_model=self.hyper_model,
                            n_timesteps=self.n_timesteps,
                            n_features=self.n_features,
                            n_outputs=self.n_outputs)

        elif self.model_name == Model.conv_lstm.value:

            model = ConvLSTM(hyper_model=self.hyper_model,
                             n_timesteps=self.n_timesteps,
                             n_features=self.n_features,
                             n_outputs=self.n_outputs)

        else:

            sys.exit(1)

        return model


class CodeBehaviorFactory():

    def __init__(self, code_behavior_name):

        self.code_behavior_name = code_behavior_name
        print("Code behavior: {}".format(self.code_behavior_name))

    def create(self):

        if self.code_behavior_name == CodeBehavior.HPT.value:

            behavior = HyperParameterTuning()

        elif self.code_behavior_name == CodeBehavior.MC.value:

            behavior = ModelsComparing()

        elif self.code_behavior_name == CodeBehavior.MR.value:

            behavior = ModelRunning()
        else:

            sys.exit(1)

        return behavior

