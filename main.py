from Classes.data import *
from ClassesML.factory import Factory
from ClassesML.models import Model
import matplotlib.pyplot as plt
import sys

from keras.utils import to_categorical

pathInfo = PathInfo()
data = Data(pathInfo=pathInfo)
X_train, y_train, X_test, y_test = data.all_data

# parameters
verbose = 0
epochs = 30
batch_size = 64

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]
n_outputs = y_train.shape[1]

factory = Factory(model_name=Model.conv_lstm,
                  n_timesteps=n_timesteps,
                  n_features=n_features,
                  n_outputs=n_outputs)
base = factory.create()
X_train, X_test = base.fit_data_on_model(X_train=X_train,
                                         X_test=X_test)

# region main loop
REPEATS = 5
scores = list()

for r in range(REPEATS):

    # Building model
    model = base.create()

    model.fit(x=X_train, y=y_train, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(x=X_test, y=y_test, batch_size=batch_size, verbose=0)
    score = accuracy * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)

m, s = np.mean(scores), np.std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# endregion









