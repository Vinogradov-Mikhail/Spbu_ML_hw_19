import numpy as np
import os
import pandas as pd

LEARNING_RATE = 0.15
REGULARISATION = 0.005
MAX_ITERATION = 15000
TARGET_COLUMN_NAME = 53
FOLD_COUNT = 5

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


def get_gradient(X, w, Y):
    # grad = -1/N * (Sum_[n] 2*(Y_i - w*X_i - b) * X_i)
    epsilon = Y - np.dot(X, w) # ] epsilon = Y_i - w*X_i - b
    N = X.shape[0]
    gradient = (-1 / N) * 2 * np.dot(epsilon, X)

    #(\gamma ||w||_2)' = \gamma * 2 * w
    # -1/N * (Sum_[n] 2*(Y_i - w*X_i - b) * X_i) + \gamma * 2 * w -- regularization
    gradient_with_regularization = gradient + REGULARISATION * 2 * w
    return gradient_with_regularization

def train_linear_regression(X, weights, Y):
    for i in range(MAX_ITERATION):

        gradient = get_gradient(X, weights, Y)

        norm = np.linalg.norm(gradient)
        gradient_dir = np.asarray([p_i / norm for p_i in gradient])
        try:
            weights -= gradient_dir * LEARNING_RATE * (1 / np.math.log(i + 2))
        except ZeroDivisionError:
            print('zero division because i == ', i)

        if np.linalg.norm(gradient) < 0.1:
            return weights
    return weights

def calc_MSE(X, weights, Y):
    return (1 / Y.shape[0]) * ((Y - np.dot(X, weights)) ** 2).sum(axis=0)

def calc_RMSE(X, weights, Y):
    return np.sqrt(calc_MSE(X, weights, Y))

def calc_R2(X, weights, Y):
    diff = Y - np.dot(X, weights)
    return 1 - ((diff ** 2).sum(axis=0) / ((Y - Y.mean()) ** 2).sum(axis=0))

def create_dataset_for_cross_validation(dataset):
    normolise_dataset = pd.DataFrame(columns=dataset.columns)

    for (columnName, columnData) in dataset.iteritems():
        if (columnName == TARGET_COLUMN_NAME):
            normolise_dataset[columnName] = columnData
        else:
            normolise_dataset[columnName] = normalization(columnData)

    (rows_count, columns_count) = normolise_dataset.shape

    #create different val sets
    test_folds = [pd.DataFrame(columns=dataset.columns) for i in range(FOLD_COUNT)]
    copy_normolise_dataset = normolise_dataset.copy()

    parts_else = FOLD_COUNT
    for i in range(FOLD_COUNT):
         temp_dataset = copy_normolise_dataset.sample(frac=1/parts_else) # frac = 1 / fold_count
         copy_normolise_dataset = copy_normolise_dataset.drop(temp_dataset.index)
         test_folds[i] = temp_dataset
         parts_else = parts_else - 1
    return normolise_dataset, test_folds

def cross_validation(fold_num, normolise_dataset, test_fold):
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

    weights = train_linear_regression(x_train, weights, y_train)

    # create out array
    train_results = np.array([])

    # calculate different metrics
    # calculate for train

    MSE = calc_MSE(x_train, weights, y_train)
    train_results = np.append(train_results, MSE)
    RMSE = calc_RMSE(x_train, weights, y_train)
    train_results = np.append(train_results, RMSE)
    R2 = calc_R2(x_train, weights, y_train)
    train_results = np.append(train_results, R2)
    print("================================", '\n')
    print("Fold #", fold_num + 1, "\n")
    print("MSE for train - ", MSE)
    print("RMSE for train - ", RMSE)
    print("R^2 for test - ", R2, "\n")

    # calculate for test

    MSE = calc_MSE(x_test, weights, y_test)
    train_results = np.append(train_results, MSE)
    RMSE = calc_RMSE(x_test, weights, y_test)
    train_results = np.append(train_results, RMSE)
    R2 = calc_R2(x_test, weights, y_test)
    train_results = np.append(train_results, R2)
    print("MSE for test - ", MSE)
    print("RMSE for test - ", RMSE)
    print("R^2 for test - ", R2, "\n")

    train_results = np.append(train_results, weights)

    return train_results

def create_csv_with_train_results(columns_count, train_results):
    csv_to_save = pd.DataFrame(columns=['', 'T1', 'T2', 'T3', 'T4', 'T5', 'E', 'STD'])

    # create first column
    first_column = np.array(['MSE-train', 'RMSE-train', 'R^2-train', 'MSE-test', 'RMSE-test', 'R^2-test'])

    for i in range(columns_count):
        first_column = np.append(first_column, ['f' + str(i)])

    csv_to_save[''] = first_column
    csv_to_save['T1'] = train_results[0]
    csv_to_save['T2'] = train_results[1]
    csv_to_save['T3'] = train_results[2]
    csv_to_save['T4'] = train_results[3]
    csv_to_save['T5'] = train_results[4]

    mean = np.array([])
    std = np.array([])
    len = first_column.shape[0]
    for i in range(len):
        mean = np.append(mean, np.mean(
            [csv_to_save['T1'][i], csv_to_save['T2'][i], csv_to_save['T3'][i], csv_to_save['T4'][i],
             csv_to_save['T5'][i]], axis=0))
        std = np.append(std, np.std(
            [csv_to_save['T1'][i], csv_to_save['T2'][i], csv_to_save['T3'][i], csv_to_save['T4'][i],
             csv_to_save['T5'][i]], axis=0))

    csv_to_save['E'] = mean
    csv_to_save['STD'] = std

    save_filename = "results_" + "LEARNING_RATE=" + str(LEARNING_RATE) + "_REGULARISATION=" + str(REGULARISATION) + "_MAX_ITERATION=" + str(MAX_ITERATION) + ".csv"
    csv_to_save.to_csv(save_filename, encoding='utf-8')

def main():
    dataset = pd.read_csv("Dataset/Features_Variant_1.csv", header=None)

    normolise_dataset = pd.DataFrame(columns=dataset.columns)

    test_folds = [pd.DataFrame(columns=dataset.columns) for i in range(FOLD_COUNT)]

    (normolise_dataset, test_folds) = create_dataset_for_cross_validation(dataset)

    (rows_count, columns_count) = normolise_dataset.shape

    train_results = [[0] for i in range(FOLD_COUNT)]
    #start cross validation
    for i in range(FOLD_COUNT):
        train_results[i] = cross_validation(i, normolise_dataset, test_folds[i])

    # save results
    create_csv_with_train_results(columns_count, train_results)

    return 0

if __name__ == "__main__":
    main()