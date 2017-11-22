import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV

if __name__ == '__main__':
    # load data set and split
    train_set = pd.read_csv('train.csv', sep=',')
    train_np = train_set.as_matrix()
    train_X = train_np[:, 1:]
    # train_X[train_X > 0] = 1
    train_y = train_np[:, 0]
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)

    test_set = pd.read_csv('test.csv', sep=',')
    test_np = test_set.as_matrix()
    # test_np[test_np > 0] = 1
    test_X = test_np
    test_y = np.zeros((test_X.shape[0],), dtype=np.int)

    # define the estimator
    clf_perceptron = linear_model.Perceptron('l2', alpha=0.0001, n_iter=30, eta0=0.1, n_jobs=2)

    # k-cv and learning curve
    # def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1., 10),
    #                         verbose=0, plot=True):
    #     """
    #     画出data在模型上的learning curve
    #     :param estimator: 分类器
    #     :param title: 表格标题
    #     :param X: 输入feature，numpy类型
    #     :param y: 输入target
    #     :param ylim: tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    #     :param cv: 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    #     :param n_jobs: 并行任务数
    #     :return: midpoint, diff
    #     """
    #     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
    #                                                             train_sizes=train_sizes, verbose=verbose)
    #     train_scores_mean = np.mean(train_scores, axis=1)
    #     train_scores_std = np.std(train_scores, axis=1)
    #     test_scores_mean = np.mean(test_scores, axis=1)
    #     test_scores_std = np.std(test_scores, axis=1)
    #
    #     if plot:
    #         plt.figure()
    #         plt.title(title)
    #         if ylim is not None:
    #             plt.ylim(*ylim)
    #         plt.xlabel('Samples Num')
    #         plt.ylabel('Accuracy')
    #         plt.gca().invert_yaxis()
    #         plt.grid()
    #
    #         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
    #                          alpha=0.1, color='b')
    #         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
    #                          alpha=0.1, color='r')
    #         plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label='Train set Accuracy')
    #         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label='Test set Accuracy')
    #
    #         plt.legend(loc='best')
    #
    #         plt.draw()
    #         plt.savefig('learning_curve.png', dpi=600)
    #         # plt.show()
    #         plt.gca().invert_yaxis()
    #
    #     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    #     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    #     return midpoint, diff
    #
    #
    # plot_learning_curve(clf_perceptron, 'Learning curve', train_X, train_y, n_jobs=1)
    # Train
    # clf_perceptron.fit(X_train, y_train)
    # acc = clf_perceptron.score(X_val, y_val)
    # print('Acc: ', acc)
    alpha_range = [0.0001, 0.0005, 0.001, 0.01, 0.1]
    iter_range = [20, 30, 35, 40, 50]
    eta_range = [0.1, 0.2, 0.4, 0.8, 1.0]
    param_grid = {'alpha': alpha_range, 'n_iter': iter_range, 'eta0': eta_range}
    gs = GridSearchCV(estimator=clf_perceptron, param_grid=param_grid, n_jobs=2)
    gs = gs.fit(train_X, train_y)
    print(gs.best_score_)
    print(gs.best_params_)

    clf_perceptron = gs.best_estimator_

    # Test
    clf_perceptron.fit(train_X, train_y)
    test_y = clf_perceptron.predict(test_X)
    id = np.arange(1, len(test_y) + 1).reshape(-1)
    result_np = np.column_stack((id, test_y))
    result_pd = pd.DataFrame(result_np, columns=['ImageID', 'Label'])
    result_pd.to_csv('result.csv', sep=',', index=False)
    # 如果测试集中提供样本ID，可以如下生成DataFrame
    # result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
