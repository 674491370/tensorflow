from sklearn.datasets import load_iris

iris_dataset = load_iris()
print('keys of iris_dataset:{}\n'.format(iris_dataset.keys()))

"""
查看数据形式
"""

import numpy as np

for key in iris_dataset.keys():
    print('current keys:"{}",key type:{}'.format(key, type(iris_dataset[key])))
    # 如果 np.ndarray 可以说明是训练数据以及对应的标签
    if isinstance(iris_dataset[key], np.ndarray):
        print(iris_dataset[key][0])
    elif isinstance(iris_dataset[key], str):
        print(iris_dataset[key][:150])
    else:
        print(iris_dataset[key])

"""
数据处理
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0,
                                                    test_size=0.3)

# 构建模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []
for n in k_range:
    knn = KNeighborsClassifier(n)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_knn = KNeighborsClassifier(n_neighbors=3)  # 选择最优的K=3传入模型
best_knn.fit(x_train, y_train)  # 训练模型
print(best_knn.score(x_test, y_test))  # 看看评分
