#!/usr/bin/env python

"""
Created by Ofek P Dec 2019
Interpreter Python 3.7

Multi class logistic regression implementation for MNIST

Example output:
0        Accuracy [9.96%]        Cost [23026.437882220947]
1        Accuracy [68.05%]       Cost [17460.64097398914]
...
2096     Accuracy [92.39%]       Cost [2691.3458081248396]
2097     Accuracy [92.39%]       Cost [2691.30577193791]
2098     Accuracy [92.39%]       Cost [2691.2657713311696]
2099     Accuracy [92.39%]       Cost [2691.2258062588553]
DONE! Elapsed time [224.71 sec]
"""

import time
import numpy as np
from keras.datasets import mnist


def count_nans(matrix):
    return np.count_nonzero(np.isnan(matrix))


def count_infs(matrix):
    return np.count_nonzero(np.isinf(matrix))


# returns matrix [train_set_size, K]
def one_hot(y, K):
    y_one_hot = np.zeros((y.size, K))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def softmax(X):
    # the sum is across the rows of WX
    X_exp = np.exp(X)
    denom = np.sum(X_exp, 0) + 1e-6
    assert(count_nans(denom) == 0)
    assert(count_infs(denom) == 0)
    return X_exp / denom


# returns a matrix of size [K, train_set_size]
def yhat(W, X):
    WX = W.dot(X.transpose())
    return softmax(WX)


# cost function
def cost(W, X, y_train_one_hot):
    # mult with y_train_one_hot is elem by elem mult
    return -np.sum(np.log(yhat(W, X) + 1e-6) * np.transpose(y_train_one_hot))


# result - [K, train_set_size] dot [train_set_size, 28 * 28 + 1] = [K, 28 * 28 + 1]
def gd(W, X, y_train_one_hot, etta):
    # yhat(W, X) - [K, train_set_size]
    # X - [train_set_size, 28 * 28 + 1]
    dW = (yhat(W, X) - np.transpose(y_train_one_hot)).dot(X)
    W = W - etta * dW
    return W


# X_test - with bias
def eval_model(W, X_test, y_test_one_hot):
    K = np.shape(y_test_one_hot)[1]  # num of classes
    correct = np.sum(one_hot(np.argmax(yhat(W, X_test), 0), K) * y_test_one_hot)
    return correct / np.shape(X_test)[0]


def main():
    max_num_of_iterations = 100000
    min_delta_cost = 0.04
    K = 10  # 10 classes [0, 1, ..., 9]

    #np.set_printoptions(threshold=sys.maxsize)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_set_size = np.shape(x_train)[0]
    test_set_size = np.shape(x_test)[0]

    # convert y_train and y_test to 1-hot representation
    y_train_one_hot = one_hot(y_train, K)
    y_test_one_hot = one_hot(y_test, K)

    # reshape 28 X 28 matrices to 1-dim vectors
    x_train = x_train.reshape(train_set_size, 28 * 28)
    x_test = x_test.reshape(test_set_size, 28 * 28)

    # last element in each row is for bias
    X = np.c_[x_train / 255, np.ones(train_set_size, dtype=int)]
    X_test = np.c_[x_test / 255, np.ones(test_set_size, dtype=int)]

    assert(count_nans(X) == 0)
    assert(count_infs(X) == 0)

    # start with a random W matrix, last element is for bias
    W = 0.0003 * np.random.rand(K, 28 * 28 + 1)

    etta = 0.00001

    prev_cost = float('Inf')
    for i in range(max_num_of_iterations):
        accuracy = eval_model(W, X_test, y_test_one_hot) * 100
        accuracy = float("{0:.2f}".format(accuracy))
        curr_cost = cost(W, X_test, y_test_one_hot)
        print(str(i) + "\t\t Accuracy [" + str(accuracy) + "%] \t\t Cost [" + str(curr_cost) + "]")
        if (abs(prev_cost - curr_cost) < min_delta_cost):
            break
        prev_cost = curr_cost
        W = gd(W, X, y_train_one_hot, etta)


start_time = time.time()
main()
elapsed_time_sec = time.time() - start_time
elapsed_time_sec = float("{0:.2f}".format(elapsed_time_sec))
print("DONE! Elapsed time [" + str(elapsed_time_sec) + " sec]")