import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data_path = r'E:\OpenSourceDatasetCode\Dataset\Titiannic Disaster'
train_set = pd.read_csv(data_path + r'\train.csv')
test_set = pd.read_csv(data_path + r'\test.csv')
train_set.head(3)
test_set.head(3)

print('***********Train*************')
print(train_set.isnull().sum())
print('***********Test*************')
print(test_set.isnull().sum())

train_set[train_set.Embarked.isnull()]
print(train_set.Embarked.value_counts())
print('*************************')
print(train_set[(train_set.Pclass == 1)].Embarked.value_counts())
train_set.Embarked.fillna('S', inplace=True)

# Cabin 为空赋值0，不为空赋值
train_set['Cabin'] = train_set['Cabin'].isnull().apply(lambda x: 'Null' if x is True else 'Not Null')
test_set['Cabin'] = test_set['Cabin'].isnull().apply(lambda x: 'Null' if x is True else 'Not Null')

# Name/Ticket 暂不考虑
del train_set['Name'], test_set['Name']
del train_set['Ticket'], test_set['Ticket']


# 以5岁为一个周期离散，同时10以下，60岁以上的年分别归类
def age_map(x):
    if x < 10:
        return '10-'
    if x < 60:
        return '%d-%d' % (x // 5 * 5, x // 5 * 5 + 5)
    elif x >= 60:
        return '60+'
    else:
        return 'Null'


train_set['Age_map'] = train_set['Age'].apply(lambda x: age_map(x))
test_set['Age_map'] = test_set['Age'].apply(lambda x: age_map(x))
# train_set.groupby('Age_map')['Survived'].agg(['count', 'mean'])

test_set[test_set.Fare.isnull()]
# 取相似特征乘客的均值
test_set.loc[test_set.Fare.isnull(), 'Fare'] = test_set[(test_set.Pclass == 3)
                                                        & (test_set.Embarked == 'S') &
                                                        (test_set.Sex == 'male')].dropna().Fare.mean()

# Fare分布太宽，做scale，加速模型收敛
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(train_set['Fare'].values.reshape(-1, 1))
train_set.Fare = fare_scale_param.transform(train_set['Fare'].values.reshape(-1, 1))
test_set.Fare = fare_scale_param.transform(test_set['Fare'].values.reshape(-1, 1))

# 将类别型变量one-hot编码
train_X = pd.concat([train_set[['SibSp', 'Parch', 'Fare']],
                     pd.get_dummies(train_set[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])], axis=1)
train_y = train_set.Survived
test_X = pd.concat([test_set[['PassengerId', 'SibSp', 'Parch', 'Fare']],
                    pd.get_dummies(test_set[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])], axis=1)

train_X.to_csv(r'eng_train_X.csv', index=None)
train_y.to_csv(r'eng_train_y.csv', header='Survived', index=None)
test_X.to_csv(r'eng_test_X.csv', index=None)
