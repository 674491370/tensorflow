from sklearn.datasets import make_classification
"""
n_samples: 生成样本数量
n_features：生成样本的特征数，特征数=n_informative+n_redundant+n_repeated
n_informative：多信息特征的个数
n_redundant：冗余信息
n_clusters_per_class：某一类别是由cluster构成的
"""
x,y = make_classification(n_samples=1000,n_features=5,n_redundant=0,n_informative=1,n_clusters_per_class=1)

x_data_train = x[:800,:]
x_data_test = x[800:,:]
y_data_train = y[:800]
y_data_test = y[800:]
from sklearn.linear_model import Perceptron
import numpy as np
#定义感知机
clf = Perceptron(fit_intercept=True,max_iter=150,shuffle=True)
#使用训练数据进行训练
clf.fit(x_data_train,y_data_train)
#得到训练结果，权重矩阵
print(clf.coef_)
#超平面的截距，此处输出为：[0.]
print(clf.intercept_)
#利用测试数据进行验证
acc = clf.score(x_data_test,y_data_test)
y_test = clf.predict(x_data_test)
print(y_test)
