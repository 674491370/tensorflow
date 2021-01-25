import tensorflow as tf


def create_graph_1():
    g1 = tf.Graph()
    with g1.as_default():
        c1 = tf.constant(10, name='c1')
        c2 = tf.constant(20)
        print("张量c1:{}".format(c1))
        print("张量c2:{}".format(c2))
        print("张量c1所在图：{}".format(c1.graph))
        print("张量c2所在图：{}".format(c2.graph))
        print("g1图：{}".format(g1))


def tensor_extract_in_graph():
    g1 = tf.Graph()
    with g1.as_default():
        c1 = tf.constant([[1], [2]], name='c1')
        c2 = tf.constant([[1, 2]], name='c2')
        v1 = tf.Variable([[1], [2]], name='v1')
        v2 = tf.Variable([[1, 2]], name='v2')
        # 矩阵计算
        mat_res_c = tf.matmul(c1, c2, name='mat_res_c')
        mat_res_v = tf.matmul(v1, v2, name='mat_res_v')
        return mat_res_c, mat_res_v


if __name__ == "__main__":
    mat_res_c, mat_res_v= tensor_extract_in_graph()
    print(mat_res_c)
    print(mat_res_v)
   