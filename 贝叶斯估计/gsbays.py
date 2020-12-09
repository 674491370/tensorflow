# 构建简单模型
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -2], [-3, -3], [-4, -4], [1, 1], [2, 2], [3, 3]])
y = np.array([1, 1, 1, 1, 2, 2, 2])
print(np.var(X, axis=0))
"""
priors: y的先验概率，此处可为[4/7,3/7]
var_smoothing: sigma_ = sigma_+var_smoothing * np.var(X, axis=0).max()
"""
bn1 = GaussianNB(priors=None, var_smoothing=1e-9)
result = bn1.fit(X, y)
print(result)

# 观察各个属性值
p1 = bn1.priors
print(p1)

# 获取各类标记对应的先验概率
cp = bn1.class_prior_
print(cp)

# 获取各类标记对应的训练样本数
cc = bn1.class_count_
print(cc)

# 获取各个类标记在各个特征上的均值
ct = bn1.theta_
print(ct)

# 获取各个类标记在各个特征上的方差
cs = bn1.sigma_
print(cs)

# 获取参数
gp = bn1.get_params(deep=True)
print(gp)

# 训练样本
bn = GaussianNB()
bn.fit(X, y, np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
print(bn)
print(bn.theta_)
print(bn.sigma_)

# 增量式训练
bn2 = GaussianNB()
bn2.partial_fit(X, y, classes=[1, 2], sample_weight=np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))

# 输出测试机预测的类标记
pr1 = bn1.predict([[-6, -6], [4, 5]])
print(pr1)

# 输出测试样本在各个类标记预测概率值
pp1 = bn1.predict_proba([[-6, -6], [4, 5]])
print(pp1)

# 测试样本映射到类标记上得到的分数
sc = bn1.score([[-6,-6],[-4,-2],[-3,-4],[4,5]],[1,1,2,2])
print(sc)
