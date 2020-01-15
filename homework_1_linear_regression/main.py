import numpy as np
import os
import pandas as pd

LEARNING_RATE = 0.15
REGULARISATION = 0.005
MAX_ITERATION = 15000
TARGET_COLUMN_NAME = 53

# x^j+1 = x^j - lambda^j * grad(F(x^j))

# block of function for features

# normalise values in selected column
def normalization(column):
    column_min = column.min()
    column_max = column.max()
    column_range = column_max - column_min
    if(column_range == 0):
        return (column - column_min)

    return (column - column_min) / column_range


def calculate_gradient(X, w, Y):
    # grad = -1/N * (Sum_[n] 2*(Y_i - w*X_i - b) * X_i)
    epsilon = Y - np.dot(X, w) # ] epsilon = Y_i - w*X_i - b
    N = X.shape[0]
    gradient = (-1 / N) * 2 * np.dot(epsilon, X)

    #(\gamma ||w||_2)' = \gamma * 2 * w
    # -1/N * (Sum_[n] 2*(Y_i - w*X_i - b) * X_i) + \gamma * 2 * w -- regularization
    gradient_with_regularization = gradient + REGULARISATION * 2 * w
    return gradient_with_regularization

def linear_regression(X, Y, w):
    for i in range(MAX_ITERATION):

        gradient = calculate_gradient(X, w, Y)

        norm = np.linalg.norm(gradient)
        gradient_dir = np.asarray([p_i / norm for p_i in gradient])
        try:
            w -= gradient_dir * LEARNING_RATE * (1 / np.math.log(i + 2))
        except ZeroDivisionError:
            print('zero division because i == ', i)

        if np.linalg.norm(gradient) < 0.1:
            return w
    return w

def calc_MSE(x,y,weights):
    return (1 / y.shape[0]) * ((y - np.dot(x, weights)) ** 2).sum(axis=0)

def calc_RMSE(x,y,weights):
    return np.sqrt(calc_MSE(x,y,weights))

def calc_R2(X, Y, w):
    diff = Y - np.dot(X, w)
    return 1 - ((diff ** 2).sum(axis=0) / ((Y - Y.mean()) ** 2).sum(axis=0))

def fold_step(fold_num, normolise_dataset, test_fold):
    # Split for train and val as 70% - train and 30 % - val
    train_dataset = normolise_dataset.drop(test_fold.index)
    test_dataset = test_fold

    (rows_train_count, columns_train_count) = train_dataset.shape
    (rows_val_count, columns_val_count) = test_dataset.shape

    # move to x + b matrix and y
    drop_train_dataset = train_dataset.drop(TARGET_COLUMN_NAME, axis=1).copy()

    b = np.transpose(np.array([[1] * rows_train_count]))
    x_train = drop_train_dataset.values
    x_train = np.concatenate((x_train, b), axis=1)

    y_train = train_dataset[TARGET_COLUMN_NAME].values

    drop_val_dataset = test_dataset.drop(TARGET_COLUMN_NAME, axis=1).copy()

    b = np.transpose(np.array([[1] * rows_val_count]))
    x_test = drop_val_dataset.values
    x_test = np.concatenate((x_test, b), axis=1)

    y_test = test_dataset[TARGET_COLUMN_NAME].values

    # fill weight random values from 0 to 1
    x_train_column_count = x_train.shape[1]
    weights = np.random.rand(x_train_column_count)

    weights = linear_regression(x_train, y_train, weights)

    # calculate different metrics
    # calculate for train

    MSE = calc_MSE(x_train, y_train, weights)
    RMSE = calc_RMSE(x_train, y_train, weights)
    R2 = calc_R2(x_train, y_train, weights)
    print("================================", '\n')
    print("Fold ", fold_num + 1, "\n")
    print("MSE for train - ", MSE)
    print("RMSE for train - ", RMSE)
    print("R^2 for test - ", R2, "\n")

    # calculate for test

    MSE = calc_MSE(x_test, y_test, weights)
    RMSE = calc_RMSE(x_test, y_test, weights)
    R2 = calc_R2(x_test, y_test, weights)
    print("MSE for test - ", MSE)
    print("RMSE for test - ", RMSE)
    print("R^2 for test - ", R2, "\n")

def main():
    dataset = pd.read_csv("../Dataset/Features_Variant_1.csv", header=None)

    normolise_dataset = pd.DataFrame(columns=dataset.columns)

    for (columnName, columnData) in dataset.iteritems():
        if (columnName == TARGET_COLUMN_NAME):
            normolise_dataset[columnName] = columnData
        else:
            normolise_dataset[columnName] = normalization(columnData)

    (rows_count, columns_count) = normolise_dataset.shape

    fold_count = 5
    #create 5 different val sets
    test_folds = [pd.DataFrame(columns=dataset.columns) for i in range(fold_count)]
    copy_normolise_dataset = normolise_dataset.copy()

    parts = fold_count
    for i in range(fold_count):
         temp_dataset = copy_normolise_dataset.sample(frac=1/parts) # frac = 1 / fold_count
         copy_normolise_dataset = copy_normolise_dataset.drop(temp_dataset.index)
         test_folds[i] = temp_dataset
         parts = parts - 1

    #start 5-cross validation and learning linear regression
    for i in range(fold_count):
        fold_step(i, normolise_dataset, test_folds[i])
    return 0

if __name__ == "__main__":
    main()