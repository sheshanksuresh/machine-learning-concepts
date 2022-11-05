from re import A
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# split data into k subsets
def split_data(k, X, t):
    size = len(X)
    subset_size = size // k
    X_split = []
    t_split = []
    count = 0
    for i in range(k):
        if i == k:  # for last subset takes all the final values
            X_split.append(X[count:])
            t_split.append(t[count:])
        else:  # put the subset_size number of values into the subset
            X_split.append(X[count : count + subset_size])
            t_split.append(t[count : count + subset_size])
        count = count + subset_size

    return X_split, t_split


def training(X_train, t_train):
    # calculate the w matrix using equation w=((X^T)X)^-1(X^T)t
    w_den = np.matmul(np.transpose(X_train), X_train)  # (X^T)X
    # check the determinant of matrix to see if the matrix is nonsingular
    if np.linalg.det(w_den) != 0:
        w_den = np.linalg.inv(w_den)
    else:
        return 0
    w_num = np.matmul(np.transpose(X_train), np.transpose(t_train))  # (X^T)t
    w = np.matmul(w_den, w_num)
    return w


# calculate Y by using the equation Y = X*w
def calcY(w, X):
    Y = np.matmul(X, w)
    return Y


# calculate error using equation e = (1/N)(y-t)^T(y-t)
def calcError(Y, t):
    N = len(t)
    error = np.matmul(np.transpose(Y - t), (Y - t))  # (y-t)^T(y-t)
    error = error / N
    return error


# separates the features of the matrix into separate rows
def separate_X(X):
    newX = np.transpose(X)
    return newX


# merges separate rows of features
def merge_X(X):
    newX = np.transpose(X)
    return newX


def cross_validation(kf, X, t):
    error = 0
    # iterate through each subset being used for validation
    for train, test in kf.split(X, t):
        X_train, X_valid, t_train, t_valid = X[train], X[test], t[train], t[test]
        w = training(X_train, t_train)
        Y = calcY(w, X_valid)
        error += calcError(Y, t_valid)
    error /= kf.get_n_splits(
        X
    )  # this would be average validation error for current feature

    return error, w


def forward_stepwise_selection(K, X, t):
    # empty subset S
    S = []
    SIndex = []
    # separate features
    X_broken = separate_X(X)

    # break into subsets using KFold
    kf = KFold(K)

    # validation errors
    errors_valid = []

    functions = []

    for SFeature in range(
        len(X_broken)
    ):  # iterate to find which feature is most dominant
        # set error very high so that the first error will be taken
        lowest_error = [-1, np.inf, 1]
        for k in range(len(X_broken)):  # find the error on each feature
            # only check feature if it is not in subset S
            if not k in SIndex:
                # tempS/tempX will create a temporary subset to train features
                tempS = np.ones(len(X_broken[k]))  # w0 column
                if len(S) != 0:
                    tempS = np.vstack((tempS, S))
                tempS = np.vstack([tempS, X_broken[k]])
                tempX = merge_X(tempS)

                # outputs validation error after training
                temp_error, w = cross_validation(kf, tempX, t)
                if lowest_error[1] > temp_error:
                    lowest_error = [k, temp_error, w]
        if len(S) != 0:
            S = np.vstack([S, X_broken[lowest_error[0]]])
        else:
            S = X_broken[lowest_error[0]]
        SIndex.append(lowest_error[0])
        errors_valid.append(lowest_error[1])
        functions.append(lowest_error[2])
    return SIndex, errors_valid, functions


def main():
    # Load boston housing dataset and supress deprecation warning to reduce terminal clutter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X, t = load_boston(return_X_y=True)

    # X = np.array([[9.,20.,30.],[3.,3.,3.],[23.,19.,17.],[4.,5.,6.],[3.,8.,2.],[4.,0.,22.],[7.,8.,9.],[11.,2.,3.],[15.,2.,45.],[4.,7.,8.]])
    # t = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])

    k_max = 13
    K = 3  # total number of subsets for cross validation
    np.random.seed(2679)

    # split data into training and testing data sets
    X_train, X_test, t_train, t_test = train_test_split(
        X, t, test_size=0.2, shuffle=True
    )

    # get S, validation errors, and ws using forward stepwise selection
    S, errors_valid, Ws = forward_stepwise_selection(K, X_train, t_train)

    # now collect test errors
    errors_test = []
    X_broken = separate_X(X_test)
    X1 = np.ones(len(t_test))

    for index in range(len(Ws)):
        X1 = np.vstack([X1, X_broken[S[index]]])
        X = merge_X(X1)
        Y = calcY(Ws[index], X)
        error_test = calcError(Y, t_test)
        errors_test.append(error_test)

    print("S:", S)
    print("errors_valid:", errors_valid)
    print("errors_test:", errors_test)
    # print("w:", Ws)

    return


if __name__ == "__main__":
    main()
