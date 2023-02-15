import numpy as np
import pandas as pd
from Classes.information import PathInfo

path_info = PathInfo()
features = list()
with open(path_info.path_data + "features.txt") as f:
    features = [line.split()[1] for line in f.readlines()]
print("No of features: {}".format(len(features)))

# region GET TRAIN
# get the data from txt files to pandas dataframe
X_train = pd.read_csv(path_info.path_data + "train\\X_train.txt", delim_whitespace=True, header=None)
X_train.columns = [features]

# add subject column to the dataframe
X_train["subject"] = pd.read_csv(path_info.path_data + "train\\subject_train.txt", header=None, squeeze=True)

y_train = pd.read_csv(path_info.path_data + "train\\y_train.txt", names=["Activity"], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
                              4:'SITTING', 5:'STANDING', 6:'LAYING'})

# put all columns in a single dataframe
train = X_train
train['Activity'] = y_train
train['ActivityName'] = y_train_labels
print(train.sample(2))
print("train shape: {}".format(train.shape))
# endregion

# region GET TEST
# get the data from txt files to pandas dataflame
X_test = pd.read_csv(path_info.path_data + "test\\X_test.txt", delim_whitespace=True, header=None)
X_test.columns = [features]

# add subject column to the dataframe
X_test["subject"] = pd.read_csv(path_info.path_data + "test\\subject_test.txt", header=None, squeeze=True)

# get y labels from the txt file
y_test = pd.read_csv(path_info.path_data + "test\\y_test.txt", names=["Activity"], squeeze=True)
y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
                            4:'SITTING', 5:'STANDING', 6:'LAYING'})

# put all columns in a single dataframe
test = X_test
test['Activity'] = y_test
test['ActivityName'] = y_test_labels
print(test.sample(2))
print("test shape: {}".format(test.shape))
# endregion

# check for duplicates
print('No of duplicates in train: {}'.format(sum(train.duplicated())))
print('No of duplicates in test : {}'.format(sum(test.duplicated())))

# checking for NaN/null values
print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))

# save this dataframe in a csv files
train.to_csv(path_info.path_csv + 'train.csv', index=False)
test.to_csv(path_info.path_csv + 'test.csv', index=False)