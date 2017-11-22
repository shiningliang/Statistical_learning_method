import pandas as pd
import numpy as np
import random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def load_data():
    train_X = pd.read_csv('eng_train_X.csv', sep=',')
    train_y = pd.read_csv('eng_train_y.csv', sep=',')
    test_X = pd.read_csv('eng_test_X.csv', sep=',')

    return train_X, train_y, test_X




if __name__ == '__main__':
    train_X, train_y, test_X = load_data()
    train_X = train_X.values
    train_y = train_y.values.reshape(train_y.shape[0])
    test_Id = test_X['PassengerId'].values
    test_X = test_X.as_matrix()[:, 1:]

    clf = AdaBoostClassifier()
    estimators = [50, 100, 150, 200]
    lr = [0.4, 0.6, 0.8, 1.0]
    param_grid = {'n_estimators': estimators, 'learning_rate': lr}
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=4)
    gs = gs.fit(train_X, train_y)
    print(gs.best_score_)
    print(gs.best_params_)
    clf_ada = gs.best_estimator_

    test_y = clf_ada.predict(test_X).astype(int)
    submission = pd.DataFrame({
        "PassengerId": test_Id,
        "Survived": test_y
    })
    submission.to_csv(r'submission_eng.csv', index=False)
