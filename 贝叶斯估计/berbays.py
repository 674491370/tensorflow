import numpy as np

from sklearn.naive_bayes import BernoulliNB

X = np.array([[1, 2, 3, 3], [1, 3, 4, 4], [2, 4, 5, 5]])
y = np.array([1, 2, 3])
"""
alpha: 平滑系数
binarize: 将特征二值化的阈值
fit_prior： 使用数据拟合先验概率
"""
clf = BernoulliNB(alpha=2.0, binarize=3.0, fit_prior=True)
clf.fit(X, y)
print("class_prior:", clf.class_prior)
print("class_count_:", clf.class_count_)  # 按类别顺序输出其对应个数
print("class_log_prior_:", clf.class_log_prior_)  # 先验概率对数值
print("feature_count_:", clf.feature_count_)  # 各类别个特征之和
print("n_features_:", clf.n_features_)
print("feature_log_prob_:", clf.feature_log_prob_)  # 指定类的各特征的条件概率的对数
# 其他参数与方法与MultinomialNB类似
