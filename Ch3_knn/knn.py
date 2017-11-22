import pandas as pd
import numpy as np
import time
import heapq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

K = 10


def linear_KNN(test_data, train_data, train_label):
    test_label = []
    for test_sample in test_data:
        dists = np.linalg.norm(test_sample - train_data, axis=1)
        # argsort() returns the indices that would sort an array in a ascending order
        sorted_dists_index = np.argsort(dists)
        class_count = np.zeros((10,), dtype=np.int)
        for i in range(10):
            class_count[train_label[sorted_dists_index[i]]] += 1

        class_count = list(class_count)
        max_count = max(class_count)
        test_label.append(class_count.index(max_count))

    return np.array(test_label)


def heap_KNN(test_data, train_data, train_label):
    test_label = []
    # count = 1
    for test_sample in test_data:
        # print(count)
        tmp_heap = TopK_Heap()
        dists = np.linalg.norm(test_sample - train_data, axis=1)
        for dist, label in zip(dists, train_label):
            tmp_heap.push(Element(dist, label))

        top_k = tmp_heap.TopK()
        top_k_label = [elem[1] for elem in top_k]
        test_label.append(Counter(top_k_label).most_common(1)[0][0])
        # count += 1

    return np.array(test_label)


class TopK_Heap(object):
    def __init__(self, initial=None):
        self.k = K
        self._data = []

    def push(self, elem):
        if len(self._data) < self.k:
            heapq.heappush(self._data, (elem.dist, elem.label))
        else:
            topk_small = list(self._data[0])
            if elem.dist > topk_small[0]:
                heapq.heapreplace(self._data, (elem.dist, elem.label))

    def TopK(self):
        if (len(self._data) > 1):
            return [heapq.heappop(self._data) for x in range(len(self._data))]
        else:
            return None


class Element():
    def __init__(self, dist, label):
        self.dist = -dist
        self.label = label


if __name__ == '__main__':
    # load data set and split
    print('Loading data')
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
    start = time.time()
    # y_pre = linear_KNN(X_val, X_train, y_train)
    y_pre = linear_KNN(X_val, X_train, y_train)
    test_y = y_pre
    end = time.time()
    print('Predicting costs ', end - start, 'seconds')
    acc = accuracy_score(y_val, y_pre)
    print('The accuracy score is ', acc)
    # id = np.arange(1, len(test_y) + 1).reshape(-1)
    # result_np = np.column_stack((id, test_y))
    # result_pd = pd.DataFrame(result_np, columns=['ImageID', 'Label'])
    # result_pd.to_csv('result.csv', sep=',', index=False)
