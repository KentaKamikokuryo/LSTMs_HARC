import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix


class Utilities:

    def __init__(self):

        sns.set(style='ticks', rc={"grid.linewidth": 0.1})
        sns.set_context("paper", font_scale=1)
        color = sns.color_palette("Set2", 6)
        plt.rcParams['font.family'] = 'MS Gothic'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

    @staticmethod
    def plot_acc_for_dl(history):

        plt.ioff()
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(history.history['acc'])
        ax.plot(history.history['val_acc'])
        plt.title('Model accuracy', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylim(0, 1)
        plt.legend(['Train', 'Test'], loc='upper left', fontsize=18)

        return fig

    @staticmethod
    def plot_loss_for_dl(history):

        plt.ioff()
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.legend(['Train', 'Test'], loc='upper left')

        return fig

    @staticmethod
    def plot_imbalance(y1: np.ndarray, y2: np.ndarray, y1_title: str = "", y2_title: str = "", whole_title: str=""):

        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        autopct = "%.2f"
        pd.Series(y1).value_counts().plot.pie(autopct=autopct, ax=axs[0])
        axs[0].set_title(y1_title)
        axs[0].set_ylabel("Labels [%]")
        pd.Series(y2).value_counts().plot.pie(autopct=autopct, ax=axs[1])
        axs[1].set_title(y2_title)
        axs[1].set_ylabel("Labels [%]")
        fig.suptitle(whole_title)
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_imbalance_one(y1: np.ndarray, y1_title: str = "", whole_title: str = ""):

        fig, axs = plt.subplots(ncols=1, figsize=(10, 5))
        autopct = "%.2f"
        pd.Series(y1).value_counts().plot.pie(autopct=autopct, ax=axs[0])
        axs[0].set_title(y1_title)
        axs[0].set_ylabel("Labels [%]")
        fig.suptitle(whole_title)
        fig.tight_layout()

        return fig

    @staticmethod
    def show_values(pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857
        By HYRY
        '''

        pc.update_scalarmappable()
        ax = pc.axes
        # ax = pc.axes# FOR LATEST MATPLOTLIB
        # Use zip BELOW IN PYTHON 3
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    @staticmethod
    def cm2inch(*tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    @staticmethod
    def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
                correct_orientation=False, cmap='RdBu', fmt="%.2f"):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()
        # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, AUC.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        Utilities.show_values(c, fmt=fmt)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

        plt.xticks(rotation=20)
        plt.yticks(rotation=45)

            # resize
        fig = plt.gcf()
        # fig.set_size_inches(cm2inch(40, 20))
        # fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(Utilities.cm2inch(figure_width, figure_height))

        return fig

    @staticmethod
    def plot_classification_report(classification_report, title="Classification report", cmap="BuPu"):
        """
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        :param classification_report:
        :param title:
        :param cmap:
        :return:
        """

        lines = classification_report.split("\n")
        classes = []
        plotMat = []
        support = []
        class_names = []

        for line in lines[2: (len(lines) - 5)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [x for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            print(v)
            plotMat.append(v)

        print('plotMat: {0}'.format(plotMat))
        print('support: {0}'.format(support))

        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = True
        fig = Utilities.heatmap(np.array(plotMat),
                                title,
                                xlabel,
                                ylabel,
                                xticklabels,
                                yticklabels,
                                figure_width,
                                figure_height,
                                correct_orientation,
                                cmap=cmap)

        return fig

    @staticmethod
    def plot_bar_classification_scores(ldf: list):

        frame_scores = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), ldf).T
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        frame_scores.plot.bar(ax=ax, cmap="RdYlBu", edgecolor="black")
        plt.xticks(rotation=45)
        ax.legend(loc="best")
        ax.set_ylabel("Score")
        ax.set_xlabel("Models")
        ax.set_title("Cross validation model benchmark")

        return fig

    @staticmethod
    def plot_permutation_importance(ML_model, X, y, df):

        test_result = permutation_importance(estimator=ML_model,
                                             X=X,
                                             y=y,
                                             n_repeats=10)

        df_importance = pd.DataFrame(zip(df.columns, test_result["importances"].mean(axis=1)),
                                     columns=["Features", "Importance"])
        df_importance = df_importance.sort_values("Importance", ascending=False)

        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)
        sns.barplot(x="Importance", y="Features", data=df_importance, ci=None, ax=ax)
        plt.title("Permutation Importance")
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, index, model_name):

        plt.ioff()
        fig = plt.figure(figsize=(36, 36))
        ax = fig.add_subplot(1, 1, 1)

        matrix = confusion_matrix(y_test, y_pred)

        xlabel = 'Predicted Class'
        ylabel = 'True Class'
        title = "Confusion matrix - " + str(model_name)
        xticklabels = index
        yticklabels = index
        figure_width = 25
        figure_height = len(index) + 7
        correct_orientation = True
        fig = Utilities.heatmap(np.array(matrix),
                                title,
                                xlabel,
                                ylabel,
                                xticklabels,
                                yticklabels,
                                figure_width,
                                figure_height,
                                correct_orientation,
                                cmap="BuPu",
                                fmt="%.f")

        # matrix = confusion_matrix(y_test, y_pred)
        # matrix = pd.DataFrame(matrix, columns=index, index=index)
        # sns.heatmap(matrix, annot=True, fmt="d", cmap="cool", cbar=True, ax=ax)

        return fig

    @staticmethod
    def plot_heatmap_corr_matrix(corr_matrix):

        plt.ioff()
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        sns.heatmap(corr_matrix, cmap="BuPu", vmin=-1, vmax=1, annot=True, ax=ax)

        plt.title("Confusion matrix showing feature correlations")

        return fig

    @staticmethod
    def save_figure(fig, path: str, figure_name: str, close_figure: bool = True):

        fig.savefig(path + figure_name + ".png")
        print("Figure saved to: " + path + figure_name + ".png")

        if close_figure:
            plt.close()

    @staticmethod
    def plot_latent_space(X_train, y_train=None, X_test=None, y_test=None, is_save=False, path_save=None):

        if is_save:

            plt.ioff()

        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(1, 1, 1)

        if (X_test is not None) and (y_test is not None):

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Utilities.create_colors(unique_class=unique_class)

            for num in unique_class:
                bool_list_train = [y_train == num]
                bool_list_train = np.array(bool_list_train)
                bool_list_train = bool_list_train.reshape(-1)

                component1_train = X_train[:, 0]
                component2_train = X_train[:, 1]
                component1_train = component1_train[bool_list_train]
                component2_train = component2_train[bool_list_train]

                ax.scatter(component1_train, component2_train, color=colors[num], label=str(num) + '_train', s=20,
                           marker='o', ec='none')

            for num in unique_class:
                bool_list_test = [y_test == num]
                bool_list_test = np.array(bool_list_test)
                bool_list_test = bool_list_test.reshape(-1)

                component1_test = X_test[:, 0]
                component2_test = X_test[:, 1]
                component1_test = component1_test[bool_list_test]
                component2_test = component2_test[bool_list_test]

                ax.scatter(component1_test, component2_test, color=colors[num], label=str(num) + '_test', s=20,
                           marker='o', ec='k', lw=0.5)

                ax.legend(fontsize=8)

        elif y_train is None:

            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, len(X_train)))

            component1_train = X_train[:, 0]
            component2_train = X_train[:, 1]

            ax.scatter(component1_train, component2_train, color=colors, s=20, marker='o', ec='k', lw=0.5)

        else:

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Utilities.create_colors(unique_class=unique_class)

            for num in unique_class:
                bool_list = [y_train == num]
                bool_list = np.array(bool_list)
                bool_list = bool_list.reshape(-1)

                component1 = X_train[:, 0]
                component2 = X_train[:, 1]
                component1 = component1[bool_list]
                component2 = component2[bool_list]

                ax.scatter(component1, component2, color=colors[num], label=str(num), s=20, marker='o', ec='none', lw=0.5)

                ax.legend(fontsize=8)

        ax.set_xlabel('1st_component')
        ax.set_ylabel('2nd_component')

        if is_save and (path_save is not None):

            plt.savefig(path_save)
            plt.close()

        else:
            plt.show()

        return fig, ax

    @staticmethod
    def set_plot_latent_space(fig, ax,
                              X_train, y_train=None, X_test=None, y_test=None,
                              is_save=False, path_save=None,
                              id=None, should_annotate=True):

        if is_save:
            plt.ioff()

        if (X_test is not None) and (y_test is not None):

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Utilities.create_colors(unique_class=unique_class)

            for num in unique_class:

                bool_list_train = [y_train == num]
                bool_list_train = np.array(bool_list_train)
                bool_list_train = bool_list_train.reshape(-1)

                component1_train = X_train[:, 0]
                component2_train = X_train[:, 1]
                component1_train = component1_train[bool_list_train]
                component2_train = component2_train[bool_list_train]

                ax.plot(component1_train, component2_train, color=colors[num], label=str(num) + '_train',
                        linestyle="None", markersize=20, marker='o', markeredgecolor='none')

                if should_annotate:

                    true_indices_train = [str(i) for i, bool_value in enumerate(bool_list_train) if bool_value]

                    for i, index in enumerate(true_indices_train):

                        ax.text(component1_train[i], component2_train[i], index, color=colors[num], fontsize=8.0, fontweight=4)

            for num in unique_class:

                bool_list_test = [y_test == num]
                bool_list_test = np.array(bool_list_test)
                bool_list_test = bool_list_test.reshape(-1)

                component1_test = X_test[:, 0]
                component2_test = X_test[:, 1]
                component1_test = component1_test[bool_list_test]
                component2_test = component2_test[bool_list_test]

                ax.plot(component1_test, component2_test, color=colors[num], label=str(num) + '_test',
                        linestyle="None", markersize=20, marker='o', markeredgecolor='k', markeredgewidth=0.5)

                ax.legend(fontsize=12, loc="upper left")

                if should_annotate:

                    true_indices_test = [str(i) for i, bool_value in enumerate(bool_list_test) if bool_value]

                    for i, index in enumerate(true_indices_test):

                        ax.text(component1_test[i], component2_test[i], index, color=colors[num], fontsize=8.0, fontweight=4)

        elif y_train is None:

            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, len(X_train)))

            component1_train = X_train[:, 0]
            component2_train = X_train[:, 1]

            ax.plot(component1_train, component2_train, color=colors,
                    linestyle="None", markersize=20, marker='o', markeredgecolor='k', markeredgewidth=0.5)

        else:

            unique_class = np.unique(y_train)  # make unique_class ascending order automatically
            colors = Utilities.create_colors(unique_class=unique_class)

            for i, num in enumerate(unique_class):

                bool_list = [y_train == num]
                bool_list = np.array(bool_list)
                bool_list = bool_list.reshape(-1)

                component1 = X_train[:, 0]
                component2 = X_train[:, 1]
                component1 = component1[bool_list]
                component2 = component2[bool_list]

                ax.plot(component1, component2, color=colors[num], label=str(num),
                        linestyle="None", markersize=5, marker='o', markeredgecolor='none', markeredgewidth=0.)

                if should_annotate:

                    true_indices = [str(i) for i, bool_value in enumerate(bool_list) if bool_value]

                    for i, index in enumerate(true_indices):

                        # ax.text(component1[i], component2[i], index, color=colors[num], fontsize=8.0, fontweight=4)
                        ax.text(component1[i], component2[i], id[int(index)], color=colors[num], fontsize=8.0, fontweight=4)

            ax.legend(fontsize=12, loc="upper left")

        ax.set_xlabel('1st_component')
        ax.set_ylabel('2nd_component')
        ax.set_aspect('equal')

        if is_save and (path_save is not None):

            plt.savefig(path_save)
            plt.close()

        else:
            plt.show()

    @staticmethod
    def create_colors(unique_class, cmap_name="tab20"):

        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, len(unique_class)))
        colors = dict(zip(unique_class, colors))

        return colors