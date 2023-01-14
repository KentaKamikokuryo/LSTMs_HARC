from Classes.data import *
from Classes.ranked import Ranked
from ClassesML.hyperparameters import HyperParameters
from ClassesML.factory import Factory, CodeBehaviorFactory
from ClassesML.models import Model, ModelInfo
from ClassesML.behaviors import CodeBehavior, IBehavior
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import logging as log
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


def output_score(metric, y_true, y_pred):

    if metric == "accuracy":

        score = metrics.accuracy_score(y_true, y_pred)

    elif metric == "f1":

        score = metrics.f1_score(y_true, y_pred, average="weighted")

    elif metric == "recall":

        score = metrics.recall_score(y_true, y_pred, average="weighted")

    elif metric == "roc_auc":

        score = metrics.roc_auc_score(y_true, y_pred, average="weighted")

    elif metric == "precision":

        score = metrics.precision_score(y_true, y_pred, average="weighted")

    else:

        log.error("metric is not included!!!!!!")
        sys.exit(1)

    return score


class Manager:

    def __init__(self, interface: IBehavior):

        # Set the interface (code behavior)
        self.interface = interface

        self.metrics = ["accuracy", "f1", "recall", "precision"]

        self.metric = self.metrics[1]

        self.pathInfo = PathInfo()

        self.n_K_fold = 5

        self.verbose = 0

        # self.REPEATS = 4

        # Set the model name for validating
        self.model_info = ModelInfo()

        self._set_data()

    def _set_data(self):

        # generate data folder
        data_folder = Data(pathInfo=self.pathInfo)

        # train data will be processed with one-hot vector
        self.X_train, self.y_train, self.X_test, self.y_test = data_folder.all_data
        self.subject_train, self.subject_test = data_folder.subject_train_test
        self.class_name = data_folder.class_names

    def _set_hyper(self, model_name):

        self.ranked = Ranked(model_name=model_name,
                             metric=self.metric,
                             path=self.path_search_model)

        if self.interface.model_search_mode():

            hyper_params = HyperParameters(model_name=model_name)
            self.hyper_model_list, self.hyper_model_dict = hyper_params.generate_hyper_model(display=True)

        else:

            self.ranked.load_ranked_list(display=True)
            self.ranked.load_best_hyperparameter()
            self.hyper_model_best = self.ranked.hyperparameter_best
            self.hyper_model_list_sorted = self.ranked.hyperparameters_list_sorted

    @staticmethod
    def _generate_model_base(hyper_model, X_fit, y_fit):

        n_timesteps = X_fit.shape[1]
        n_features = X_fit.shape[2]
        n_outputs = y_fit.shape[1]

        factory = Factory(hyper_model=hyper_model,
                          n_timesteps=n_timesteps,
                          n_features=n_features,
                          n_outputs=n_outputs)
        base = factory.create()

        return base

    def _fit_valid(self, hyper_model):

        sgkf = StratifiedGroupKFold(n_splits=self.n_K_fold)

        scores = []

        for i, (fit_index, valid_index) in enumerate(sgkf.split(self.X_train, self.y_train, groups=self.subject_train)):

            print(f"Fold {i}:")

            X_fit, X_valid = self.X_train[fit_index], self.X_train[valid_index]
            y_fit, y_valid = self.y_train[fit_index], self.y_train[valid_index]

            y_fit = to_categorical(y_fit)
            # y_valid = to_categorical(y_valid)

            model_base = self._generate_model_base(hyper_model=hyper_model,
                                                   X_fit=X_fit,
                                                   y_fit=y_fit)

            X_fit, X_valid = model_base.fit_data_on_model(X_train=X_fit,
                                                          X_test=X_valid)

            dl_model = model_base.create()

            dl_model.fit(x=X_fit, y=y_fit,
                         epochs=hyper_model["epoch"],
                         batch_size=hyper_model["batch_size"],
                         verbose=self.verbose)

            y_pred = dl_model.predict(X_valid)
            y_pred = np.argmax(y_pred, axis=1)
            y_valid = y_valid.flatten()

            metric_score = output_score(self.metric, y_true=y_valid, y_pred=y_pred)
            scores.append(metric_score)

        metric_score_mean = np.mean(scores)
        metric_score_std = np.std(scores)

        return metric_score_mean, metric_score_std

    def _test(self, hyper_model):

        model_base = self._generate_model_base(hyper_model=hyper_model,
                                               X_fit=self.X_train,
                                               y_fit=self.y_train)

        X_train, X_test = model_base.fit_data_on_model(X_train=self.X_train,
                                                       X_test=self.X_test)

        self.y_train = to_categorical(self.y_train)
        # self.y_test = to_categorical(self.y_test)

        scores = []

        dl_model = model_base.create()

        dl_model.fit(x=X_train, y=self.y_train,
                     epochs=hyper_model["epoch"],
                     batch_size=hyper_model["batch_size"],
                     verbose=self.verbose)

        y_pred = dl_model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        self.y_test = self.y_test.flatten()

        score = output_score(metric=self.metric, y_true=self.y_test, y_pred=y_pred)
        scores.append(score)

        scores_all_metric = []
        for metric in self.metrics:

            score_temp = output_score(metric=metric, y_true=self.y_test, y_pred=y_pred)
            scores_all_metric.append(score_temp)

        tmp = pd.DataFrame({hyper_model["model_name"]: scores_all_metric}, index=self.metrics)
        self._ldf.append(tmp)

        metric_score_mean = np.mean(scores)
        metric_score_std = np.std(scores)

        # Plot area

        plot_model(dl_model, to_file=self.path_figure_model + hyper_model["model_name"] + "_model.png",
                   show_shapes=True, show_layer_names=True)



        return metric_score_mean, metric_score_std

    def _run_search(self):

        for model_name in self.model_info.model_names:

            self.path_search_model = self.pathInfo.set_path_search_model(model_name=model_name.value)
            self.path_figure_model = self.pathInfo.set_path_figure_model(model_name=model_name.value)
            self._set_hyper(model_name=model_name)

            for hyper_model in self.hyper_model_list:

                metric_mean, metric_std = self._fit_valid(hyper_model=hyper_model)
                self.ranked.add(hyperparameter=hyper_model, mean=metric_mean, std=metric_std)

            self.ranked.ranked(display=True, save=self.interface.save_best_search())
            self.ranked.save_best_hyperparameter()

            print("Run search is done with model: {}".format(model_name))

    def _run_comparison(self):

        ranked_comparison = Ranked(model_name="Comparison_models",
                                   metric=self.metric,
                                   path=self.pathInfo.path_results)

        self._ldf = []

        for model_name in self.model_info.model_names:

            self.path_search_model = self.pathInfo.set_path_search_model(model_name=model_name.value)
            self.path_figure_model = self.pathInfo.set_path_figure_model(model_name=model_name.value)
            self._set_hyper(model_name=model_name)

            metric_mean, metric_std = self._test(hyper_model=self.hyper_model_best)

            ranked_comparison.add(hyperparameter=self.hyper_model_best,
                                  mean=metric_mean,
                                  std=metric_std)

        ranked_comparison.ranked(display=True, save=self.interface.save_best_comparison())
        ranked_comparison.save_best_hyperparameter()

    def run(self):

        if self.interface.model_search_mode():

            self._run_search()

        else:

            self._run_comparison()


def main():

    code_behaviors = [CodeBehavior.HPT, CodeBehavior.MC, CodeBehavior.MR]
    Is = code_behaviors[:2]

    for I in Is:

        code_behavior_fac = CodeBehaviorFactory(code_behavior_name=I.value)
        interface = code_behavior_fac.create()

        manager = Manager(interface=interface)
        manager.run()


if __name__ == "__main__":

    main()









