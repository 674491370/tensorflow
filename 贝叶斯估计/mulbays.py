import numpy as np
from sklearn.naive_bayes import MultinomialNB

X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5], [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0, fit_prior=True)
clf.fit(X, y)
print(clf.class_log_prior_)
print(np.log(2/6), np.log(1/6), np.log(2/6), np.log(1/6))

# intercept_: 将多项式朴素贝叶斯解释的class_log_prior映射为线性，其值和class_log_propr相同
print(clf.class_log_prior_)
print(clf.intercept_)

# feature_log_prob_: 指定类的个特征概率(条件概率)对数值，返回形状为（n_classes,n_features）
print(clf.feature_log_prob_)
print(
    np.log((1+1+2)/(1+2+3+4+1+3+4++4+4*2))
)

# coef_: 将多项式朴素贝叶斯解释feature_log_prob映射成线性模型，其值和feature_log_prob相同
print(clf.coef_)

# class_count: 训练样本中各类别对应的样本数，按类别的顺序排序输出
print(clf.class_count_)

# feature_count_: 各类别各个特征出现的次数，返回形状为(n_classes,n_features数组)
print(clf.feature_count_)

# fit(X,y,sample_weight=None)

# partial_fit(X,y,classes=None,sample_weight=None):对于数据量大时，提供增量式训练，在线学习模型参数，参数X可以是类似数组或稀疏矩阵，在第一次调用函数，必须制定classes参数，随后调用时可以忽略

# predict(X): 在测试集上预测，输出X对应目标

# predict_log_proba(X)：测试样本划分到各个类的概率对数值

# predict_proba(X)：输出测试样本划分到各个类别的概率值

# score(X, y, sample_weight=None)：输出对测试样本的预测准确率的平均值
