from random import seed, randrange
from csv import reader
from math import sqrt
import numpy as np


# Load a CSV file
def load_csv(filename):
    data_set = list()
    head = True
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if head:
                head = False
                continue
            data_row = [float(data) for data in row[0].split(';')]
            data_set.append(data_row)

    return data_set


# Find the min and max values for each column
def dataset_minmax(data_set):
    minmax = list()
    for i in range(len(data_set[0])):
        col_values = [row[i] for row in data_set]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])

    return minmax


# Rescale dataset columns to range 0-1
def rescale_dataset(data_set, minmax):
    for row in data_set:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(data_set, k_folds):
    data_set_split = list()
    data_set_copy = list(data_set)
    fold_size = int(len(data_set) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_set_copy))
            fold.append(data_set_copy.pop(index))

        data_set_split.append(fold)

    return data_set_split


# Calculate root mean squared error
def rmse_metric(label, predicted):
    sum_error = 0.0
    for i in range(len(label)):
        prediction_error = predicted[i] - label[i]
        sum_error += (prediction_error ** 2)

    mean_error = sum_error / float(len(label))
    return sqrt(mean_error)


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data_set, algorithm, k_folds, *args):
    folds = cross_validation_split(data_set, k_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        # Train Test Predict
        predicted = algorithm(train_set, test_set, *args)
        label = [row[-1] for row in fold]
        rmse = rmse_metric(label, predicted)
        scores.append(rmse)

    return scores


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
                # print(l_rate, n_epoch, error)

    return coef


# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    # Train
    coef = coefficients_sgd(train, l_rate, n_epoch)
    # Test
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)

    return (predictions)


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def linear_regression_BFGS(train, test, n_epoch):
    train = np.mat(train)
    test = np.mat(test)
    sample_dim = np.shape(train)[1]
    theta = np.ones((sample_dim, 1))




# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]

    return yhat


# Linear Regression on wine quality dataset
seed(1)
# load and prepare data
filename = r'E:\OpenSourceDatasetCode\Dataset\winequality-white.csv'
dataset = load_csv(filename)

# normalize
minmax = dataset_minmax(dataset)
rescale_dataset(dataset, minmax)

# evaluate algorithm
k_folds = 5
l_rate = 0.01
n_epoch = 100
scores = evaluate_algorithm(dataset, linear_regression_sgd, k_folds, l_rate, n_epoch)
# print('Scores: %s' % scores)
print('Mean RMSE: %.5f' % (sum(scores) / float(len(scores))))
