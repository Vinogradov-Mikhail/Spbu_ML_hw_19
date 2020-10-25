import numpy
import pandas
import os.path
from scipy import sparse
from sklearn.externals import joblib
import scipy.sparse as scipy
from sklearn.model_selection import KFold


def calc_ERROR(X, Y, w, b):
    return Y - (numpy.dot(X, w) + b)

def calc_MSE(error):
    return (numpy.power(error, 2)).sum(axis=0) / error.size

def calc_RMSE(error):
    mse = calc_MSE(error)
    return numpy.sqrt(mse)

def compute_result(X, V, w, b):
    return b + X * w + 0.5 * ((X * V).power(2) - X.power(2) * V.power(2)).sum(axis=1)


def train(x_train, y_train, step, batch_size, k, epochs_count):
    rows_n = x_train.shape[0]
    columns_n = x_train.shape[1]

    V = scipy.csr_matrix(numpy.random.normal(size=(columns_n, k)) * 1e-15)

    b = 0
    weights = numpy.random.normal(size=(columns_n, 1)) * 1e-15

    batches_count = rows_n // batch_size + 1
    rmse = 0

    for cur_epoch in range(epochs_count):
        if (cur_epoch != 0 and rmse < 1.02):
            break

        print('Epoch - ',str(cur_epoch))

        i_random = numpy.arange(rows_n)
        numpy.random.shuffle(i_random)

        x_train_rand = x_train[i_random, :]
        y_train_rand = y_train[i_random]

        print(str(batches_count), ' batches')
        for batch in range(batches_count):
            if ((batch + 1) * batch_size < x_train_rand.shape[0]):
                x_batch = x_train_rand[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train_rand[batch * batch_size: (batch + 1) * batch_size]
            else:
                x_batch = x_train_rand[batch * batch_size:]
                y_batch = y_train_rand[batch * batch_size:]

            error = y_batch - compute_result(x_batch, V, weights, b)
            y_diff = -2 * error / batch_size

            x_transp = x_batch.transpose()
            weights -= x_batch.transpose().dot(y_diff) * step

            b -= y_diff.sum() * step

            xv = x_batch.dot(V)
            arg1 = xv.multiply(y_diff.reshape(-1, 1))
            arg1 = x_transp.dot(arg1)

            x_transp_power_2 = x_transp.power(2)
            subarg_csr = scipy.csr_matrix(x_transp_power_2.dot(y_diff))
            arg2 = V.multiply(subarg_csr)

            V -= (arg1 - arg2) * step

            if (batch % 500 == 0):
                rmse = calc_RMSE(error)
                print('\tCurrent batch:',str(batch))
                print('\t\t batch RMSE =', str(rmse))

        new_epoch = cur_epoch + 1
        step *= numpy.exp(-new_epoch * 0.01)
    return weights, b, V

folder = 'Dataset'
def load_data(filename):
    csr = joblib.load(os.path.join(folder, filename))
    return csr.astype(numpy.float32)

def load_data_target(filename):
    target = joblib.load(os.path.join(folder, filename))
    return sparse.csr_matrix(target, dtype=numpy.float32)

def main():
    print('Loading training data')

    x_data = load_data('data')
    y_data = load_data_target('target')

    print('Training data loaded')

    i = 1;
    train_rmse = []
    for n_train, n_test in KFold(n_splits=5).split(x_data):
        print('################')
        print('## Split - ', i, '##')
        print('################')
        x_train = x_data[n_train]
        y_train = y_data[n_train]
        x_test = x_data[n_test]
        y_test = y_data[n_test]

        w, b, V = train(x_train, y_train, step=1e-1, batch_size=4096, k=3, epochs_count=1)

        y_result = compute_result(x_test, V, w, b)
        error = y_result - y_test
        rmse = calc_RMSE(error)

        print('RMSE after training =',str(rmse), '+++')
        train_rmse.append(rmse)
        i = i + 1
    
    print('\n***Results table***\n')
    new_train_rmse = []
    for i in range(0, len(train_rmse)):
        new_train_rmse.append(train_rmse[i].A1[0])
        
    print(new_train_rmse)
    stats = pandas.DataFrame(numpy.vstack([new_train_rmse]), index=['Test'])
    stats = pandas.concat([stats, results.mean(axis=1).rename('Mean'), results.std(axis=1).rename('Std')], axis=1)
    print(stats)

if __name__ == "__main__":
    main()
