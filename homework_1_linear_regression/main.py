import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# x^j+1 = x^j - lambda^j * grad(F(x^j))

# block of function for features

# normalise values in selected column
def normalisation(column):
    column_min = column.min()
    column_max = column.max()
    column_range = np.math.fabs(column_max - column_min)

    return (column - column_min) / column_range

def normal_gradient_descent():
    return 0

def main():
    dataset = pd.read_csv("dataset.csv")

    normolise_dataset = pd.DataFrame(columns=dataset.columns)

    for (columnName, columnData) in dataset.iteritems():
        normolise_dataset[columnName] = normalisation(columnData)

    (rows_count, columns_count) = normolise_dataset.shape

    # Split for train and val as 2:1
    pnormolise_dataset_copy = normolise_dataset.copy()
    train_ds = pnormolise_dataset_copy.sample(frac=0.6, random_state=0)
    val_ds = pnormolise_dataset_copy.drop(train_ds.index)

    #train_ds, val_ds = train_test_split(normolise_dataset, test_size=0.3)
    print(train_ds.shape)
    print(val_ds.shape)


    return 0

if __name__ == "__main__":
    main()