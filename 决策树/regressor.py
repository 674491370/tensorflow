from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
# 交叉验证
print(cross_val_score(regressor, boston.data, boston.target, cv=10,
                      # scoring = "neg_mean_squared_error"
                      # 默认返回R平方
                      ))
