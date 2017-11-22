import pandas as pd
import numpy as np
import cv2 as cv
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()
    train_set = pd.read_csv(r'E:\PythonCode\Statistical_learning_method\train.csv', sep=',')
    train_np = train_set.as_matrix()
    train_X = train_np[:, 1:]
    # train_X[train_X > 0] = 1
    train_y = train_np[:, 0]
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)

    test_set = pd.read_csv(r'E:\PythonCode\Statistical_learning_method\test.csv', sep=',')
    test_np = test_set.as_matrix()
    # test_np[test_np > 0] = 1
    test_X = test_np
    test_y = np.zeros((test_X.shape[0],), dtype=np.int)
    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' seconds')

    print('Start training')
    prior_probability, conditional_probability = Train(train_features, train_labels)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' seconds')
