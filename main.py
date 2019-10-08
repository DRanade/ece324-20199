import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter as smooth

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 10
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
printing = True
plotting = True

# =================================== LOAD DATASET =========================================== #

######

data = pd.read_csv('data/adult.csv', delimiter=',')

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######
if printing:
    print("Data shape is originally ", data.shape)
    print("Data columns are ", data.columns)
    print("Data head is below:")
    verbose_print(data.head())
    print("Below are data labels and counts")
    print(data["income"].value_counts())

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    ######

    if printing:
        print("The number of '?' in ", feature, " are ", data[feature].isin(["?"]).sum())

    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

for feature in col_names:
    # This will give a warning and not be able to compare numbers with strings, but that's okay
    data = data[data[feature] != "?"]

if printing:
    print("After removing unknowns, data shape is now ", data.shape)

######

# =================================== BALANCE DATASET =========================================== #

######

# find out how many samples to collect from each class
numPerClass = data[data['income'] == ">50K"].shape[0]
# change data to be concatenation of equal number of samples from each class
data = pd.concat([data[data['income'] == "<=50K"].sample(numPerClass, random_state=10),
                  data[data['income'] == ">50K"].sample(numPerClass, random_state=10)]
                 )

######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

if printing:
    verbose_print(data.describe())
    pie_chart(data, 'workclass')
    pie_chart(data, 'education')
    pie_chart(data, 'marital-status')

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    ######
    if printing:
        print(data[feature].describe())

    ######

# visualize the first 3 features using pie and bar graphs

######

if printing:
    pie_chart(data, 'workclass')
    pie_chart(data, 'education')
    pie_chart(data, 'marital-status')
    binary_bar_chart(data, 'workclass')
    binary_bar_chart(data, 'education')
    binary_bar_chart(data, 'marital-status')

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
######

# these are the continuous labels
cts_labels = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# for each continuous label, normalize it
for feature in cts_labels:
    data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()

# turn into numpy array as requested
cts_data_np = data[['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']].to_numpy()

######

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
######

cat_labels = ['workclass', 'education', 'marital-status', 'occupation',
              'relationship', 'race', 'gender', 'native-country', 'income']
for feature in cat_labels:
    data[feature] = label_encoder.fit_transform(data[feature])

# At this point, incomes are done, so we can put them aside
labels = data['income'].to_numpy()
######

oneh_encoder = OneHotEncoder(sparse=False)
######

# We don't need the incomes anymore
one_hots = cat_labels[:-1]
for i in range(len(one_hots)):
    one_hots[i] = oneh_encoder.fit_transform(data[[cat_labels[i]]])
# At this point, 'one_hots' contains the onehot vectors for each feature (because a pandas dataframe can't store it)

# Let's put these one_hot vectors into a neat numpy array representing all categorical data
for i in range(len(one_hots)-1):
    one_hots[i+1] = np.concatenate((one_hots[i], one_hots[i+1]), axis=1)
cat_data_np = one_hots[-1]

# Now let's put it all together (both continuous and categorical)
inputs = np.concatenate((cts_data_np, cat_data_np), axis=1)

######
# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size=0.2, random_state=10)
# convert them into pytorch tensors for ease of use later
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
Y_train = torch.tensor(Y_train)
Y_test = torch.tensor(Y_test)

######

# =================================== LOAD DATA AND MODEL =========================================== #


def load_data(batch_size=5):

    ######

    train_loader = DataLoader(AdultDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AdultDataset(X_test, Y_test), batch_size=1, shuffle=True)

    ######
    return train_loader, val_loader


def load_model(lr=0.1):

    ######

    model = MultiLayerPerceptron()
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    total_corr = 0
    index = 0
    for X, y in val_loader:
        prediction = model(X.float())
        if prediction.item() > 0.5:
            r = 1
        else:
            r = 0
        if r == y.item():
            total_corr += 1
        index += 1

    ######

    return float(total_corr)/len(val_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    ######

    train_loader, val_loader = load_data(args.batch_size)
    model, loss_fnc, optimizer = load_model(args.lr)
    t_loss, v_loss, t_acc, v_acc = [], [], [], []

    for epoch in range(args.epochs):
        n = args.eval_every
        printI = 1
        lastN = []
        for datum, label in train_loader:
            # set up optimizer with blank gradients
            optimizer.zero_grad()
            # predict with the current weights
            predict = model(datum.float())
            # evaluate the loss for the predictions
            loss = loss_fnc(input=predict.squeeze(), target=label.float())
            # calculate the gradients per the loss function we chose
            loss.backward()
            # changes the weights one step in the right direction (for the batch)
            optimizer.step()

            if len(lastN) < n:
                lastN += [[datum, label]]
            else:
                lastN = lastN[1:] + [[datum, label]]
            if printI == n:
                t_loss += [loss.item()]
                predict = model(X_test.float())
                v_loss += [loss_fnc(input=predict.squeeze(), target=Y_test.float()).item()]
                t_acc_this = 0
                for every in range(n):
                    t_acc_this += evaluate(model, DataLoader(AdultDataset(lastN[every][0], lastN[every][1])))
                t_acc += [t_acc_this / n]
                v_acc += [evaluate(model, DataLoader(AdultDataset(X_test, Y_test)))]
                printI = 1
                print("Status update: currently on epoch", epoch, "and Loss is", loss)
                print("Number of correct predictions is ", t_acc[-1]*len(Y_train))
                print("Validation accuracy is currently: ", v_acc[-1])
            printI += 1

    if plotting:
        # Plot losses
        plt.figure()
        x = [i for i in range(len(t_loss))]
        plt.plot(x, t_loss, label='Training Loss')
        plt.plot(x, v_loss, label='Validation Loss')
        plt.title("Losses as Function of Gradient Step")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Steps")
        plt.legend()
        plt.show()

        # Plot accuracies
        x = [i for i in range(len(t_acc))]
        plt.figure()
        plt.plot(x, t_acc, label='Training Accuracy')
        plt.plot(x, v_acc, label='Validation Accuracy')
        plt.ylim(0, 1)
        plt.title("Accuracy as Function of Gradient Step")
        plt.ylabel("Accuracy")
        plt.xlabel("Gradient Steps")
        plt.legend()
        plt.show()

        # Plot losses smoothed
        plt.figure()
        x = [i for i in range(len(t_loss))]
        plt.plot(x, smooth(t_loss, 151, 3), label='Training Loss Smoothed')
        plt.plot(x, smooth(v_loss, 101, 4), label='Validation Loss')
        plt.title("Smoothed Losses as Function of Gradient Step")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Steps")
        plt.legend()
        plt.show()

        # Plot accuracies smoothed
        plt.figure()
        plt.plot(x, smooth(t_acc, 151, 5), label='Training Accuracy Smoothed')
        plt.plot(x, smooth(v_acc, 101, 4), label='Validation Accuracy Smoothed')
        plt.ylim(0, 1)
        plt.title("Smoothed Accuracy as Function of Gradient Step")
        plt.ylabel("Accuracy")
        plt.xlabel("Gradient Steps")
        plt.legend()
        plt.show()

    ######


if __name__ == "__main__":
    main()
