import numpy as np
import pandas as pd
from Classes.information import PathInfo

path_info = PathInfo()

train = pd.read_csv(path_info.path_csv + "train.csv")
test = pd.read_csv(path_info.path_csv + "test.csv")
print("train shape: {}".format(train.shape))
print("test shape: {}".format(test.shape))

columns = train.columns

# Removing '()' from column names
columns = columns.str.replace('[()]', '')
columns = columns.str.replace('[-]', '')
columns = columns.str.replace('[,]', '')

train.columns = columns
test.columns = columns

print(test.columns)

import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.7)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(16, 8))
sns.countplot(x="subject", hue="ActivityName", data=train)
plt.savefig(path_info.path_figure + "subject_activity_name.png")
plt.show()

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# perform t-sne with different perplexity values and their repective plots


def perform_tsne(X_data, y_data, perplexities, n_iter=1000, img_name_prefix="t-sne"):

    for index, perplexity in enumerate(perplexities):

        print("\nperforming t-sne with perplexity {} and with {} iterations at max".format(perplexity, n_iter))
        X_reduced = TSNE(verbose=2, perplexity=perplexity).fit_transform(X_data)
        print("Done..")

        # prepare the data for seaborn
        print("Creating plot for this t-sne visualization..")
        df = pd.DataFrame({"x": X_reduced[:, 0], "y": X_reduced[:, 1], "label": y_data})

        # draw the plot in appropriate place in the grid
        sns.lmplot(data=df, x="x", y="y", hue="label", fit_reg=False, height=8,
                   palette="Set1", markers=['^', 'v', 's', 'o', '1', '2'])
        plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('saving this plot as image in present working directory...')
        plt.savefig(path_info.path_figure + img_name)
        plt.show()
        print('Done')


X_pre_tsne = train.drop(["subject", "Activity", "ActivityName"], axis=1)
y_pre_tsne = train["ActivityName"]
perform_tsne(X_data=X_pre_tsne, y_data=y_pre_tsne, perplexities=[2, 5, 12, 20, 50])

