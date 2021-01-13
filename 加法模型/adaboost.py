import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import matplotlib.pyplot as plt
X=np.arange(10).reshape(-1,1)
y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
ada=AdaBoostClassifier(n_estimators=3,algorithm='SAMME')
ada.fit(X,y)
plt.figure(figsize=(9,6))
_=tree.plot_tree(ada[0])
plt.show()
y_=ada[0].predict(X)
# 误差率
e1=np.round(0.1*(y!=y_).sum(),4)
# 计算第一颗树的权重
# 在随机森林中每棵树的权重是相同的
# adaboost提升树中每棵树的权重不同
a1=np.round(1/2*np.log((1-e1)/e1),4)
# 样本预测正确：更新的权重
w2=0.1*np.e**(-a1*y*y_)
w2=np.round(w2/w2.sum(),4)
# 分类函数f1(x)=a1*G(x)=0.4236G1(x)

# 第二棵树
plt.figure(figsize=(9,6))
_=tree.plot_tree(ada[1])
plt.show()
# 误差率
e2=0.0714*3
a2=np.round(1/2*np.log((1-e2)/e2),4)
y_=ada[1].predict(X)
w3=w2*np.e**(-a2*y*y_)
w3=np.round(w3/w3.sum(),4)

# 第三棵树
plt.figure(figsize=(9,6))
_=tree.plot_tree(ada[2])
plt.show()
y_=ada[2].predict(X)
# 误差率
e3=(w3*(y_!=y)).sum()
a3=np.round(1/2*np.log((1-e3)/e3),4)

# 弱分类器合并成强分类器，加和
# G(x)=sign[f3(x)]=sign[a1*G1(x)+a2*G2(x)+a3*G3(x)]
y_predict=ada.predict(X)
print('y_predict=',y_predict)
y_predict1=a1*ada[0].predict(X)+a2*ada[1].predict(X)+a3*ada[2].predict(X)
y_predict1=np.sign(y_predict1)
print('y_predict1',y_predict1)