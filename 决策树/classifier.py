from sklearn.datasets import load_iris
from sklearn import tree

# 获取数据
from sklearn.model_selection import train_test_split

iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)


"""
1.criterion:string, optional (default="gini")
    (1).criterion='gini',分裂节点时评价准则是Gini指数。
    (2).criterion='entropy',分裂节点时的评价指标是信息增益。
2. max_depth: int or None, optional(default=None)。指定树的最大深度
    如果为None，表示树的深度不限。直到所有的叶子结点是纯净的，即叶子结点中
    所有眼本店属于一个类别。或者每个叶子结点包含的样本数小于min_samples_split。
3. splitter: string,optional(default="best")。指定分裂结点时的策略
    （1）. splitter='best',表示最优的分裂策略
    （2）. splitter='random'表示选择最好的随机切分策略
4.min_samples_split:int, float, optional (default=2)。表示分裂一个内部节点需要的做少样本数。
    (1).如果为整数，则min_samples_split就是最少样本数。
    (2).如果为浮点数(0到1之间)，则每次分裂最少样本数为ceil(min_samples_split * n_samples)

5.min_samples_leaf: int, float, optional (default=1)。指定每个叶子节点需要的最少样本数。
    (1).如果为整数，则min_samples_split就是最少样本数。
    (2).如果为浮点数(0到1之间)，则每个叶子节点最少样本数为ceil(min_samples_leaf * n_samples)
6.min_weight_fraction_leaf:float, optional (default=0.)
    指定叶子节点中样本的最小权重。
7.max_features:int, float, string or None, optional (default=None).
    搜寻最佳划分的时候考虑的特征数量。
    (1).如果为整数，每次分裂只考虑max_features个特征。
    (2).如果为浮点数(0到1之间)，每次切分只考虑int(max_features * n_features)个特征。
    (3).如果为'auto'或者'sqrt',则每次切分只考虑sqrt(n_features)个特征
    (4).如果为'log2',则每次切分只考虑log2(n_features)个特征。
    (5).如果为None,则每次切分考虑n_features个特征。
    (6).如果已经考虑了max_features个特征，但还是没有找到一个有效的切分，那么还会继续寻找
    下一个特征，直到找到一个有效的切分为止。
8.random_state:int, RandomState instance or None, optional (default=None)
    (1).如果为整数，则它指定了随机数生成器的种子。
    (2).如果为RandomState实例，则指定了随机数生成器。
    (3).如果为None，则使用默认的随机数生成器。
9.max_leaf_nodes: int or None, optional (default=None)。指定了叶子节点的最大数量。
    (1).如果为None,叶子节点数量不限。
    (2).如果为整数，则max_depth被忽略。
10.min_impurity_decrease:float, optional (default=0.)
     如果节点的分裂导致不纯度的减少(分裂后样本比分裂前更加纯净)大于或等于min_impurity_decrease，则分裂该节点。
     加权不纯度的减少量计算公式为：
     min_impurity_decrease=N_t / N * (impurity - N_t_R / N_t * right_impurity
                        - N_t_L / N_t * left_impurity)
     其中N是样本的总数，N_t是当前节点的样本数，N_t_L是分裂后左子节点的样本数，
     N_t_R是分裂后右子节点的样本数。impurity指当前节点的基尼指数，right_impurity指
     分裂后右子节点的基尼指数。left_impurity指分裂后左子节点的基尼指数。
11.min_impurity_split:float
     树生长过程中早停止的阈值。如果当前节点的不纯度高于阈值，节点将分裂，否则它是叶子节点。
     这个参数已经被弃用。用min_impurity_decrease代替了min_impurity_split。
12.class_weight:dict, list of dicts, "balanced" or None, default=None
     类别权重的形式为{class_label: weight}
     (1).如果没有给出每个类别的权重，则每个类别的权重都为1。
     (2).如果class_weight='balanced'，则分类的权重与样本中每个类别出现的频率成反比。
     计算公式为：n_samples / (n_classes * np.bincount(y))
     (3).如果sample_weight提供了样本权重(由fit方法提供)，则这些权重都会乘以sample_weight。
13.presort:bool, optional (default=False)
    指定是否需要提前排序数据从而加速训练中寻找最优切分的过程。设置为True时，对于大数据集
    会减慢总体的训练过程；但是对于一个小数据集或者设定了最大深度的情况下，会加速训练过程。

"""
clf = tree.DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_weight_fraction_leaf=0.,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0,
    min_impurity_split=None,
    class_weight=None,
    presort=False
)
clf = clf.fit(x_train,y_train)
score = clf.score(x_test,y_test)
print("准确率：\n", score)