import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
import time

K = 10

if __name__ == '__main__':
    print('Loading data')
    # load data set and split
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

    print('Start predicting')
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=K, weights='uniform', algorithm='brute', n_jobs=1)
    start = time.time()
    clf_knn.fit(X_train, y_train)
    acc = clf_knn.score(X_val, y_val)
    print('The accuracy score is ', acc)

    # Train set CV
    # k_range = [5, 10, 15, 20]
    # param_grid = {'n_neighbors': k_range}
    # gs = GridSearchCV(estimator=clf_knn, param_grid=param_grid, n_jobs=2)
    # gs.fit(train_X, train_y)
    # print(gs.best_score_)
    # print(gs.best_params_)
    #
    # # Test set
    # clf_knn = gs.best_estimator_
    # test_y = clf_knn.predict(test_X)
    # id = np.arange(1, len(test_y) + 1).reshape(-1)
    # result_np = np.column_stack((id, test_y))
    # result_pd = pd.DataFrame(result_np, columns=['ImageID', 'Label'])
    # result_pd.to_csv('result.csv', sep=',', index=False)
    end = time.time()
    print('Predicting costs ', end - start, 'seconds')