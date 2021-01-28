import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/Library/Fonts/Songti.ttc", size=8)


def gen_datas():
    # 输入数据
    inputs = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis]
    # 噪声数据
    noise = np.random.normal(0, 0.05, inputs.shape).astype(np.float32)
    # 输出数据
    outputs = np.square(inputs) - 0.5 * inputs + noise
    # 返回数据
    return inputs, outputs


def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae", "mse"])


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1,), name='layer1'),
        tf.keras.layers.Dense(1, name='outputs')
    ])
    compile_model(model)
    return model


def display_nn_structure(model, nn_structure_path):
    model.summary()
    keras.utils.plot_model(model, nn_structure_path, show_shapes=True)


def callback_only_params(model_path):
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch'
    )
    return ckpt_callback


def tb_callback(model_path):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_path,
        histogram_freq=1
    )
    return tensorboard_callback


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch


def plot_history(history):
    # Pandas读取损失数据
    hist = pd.DataFrame(history.history)
    # 提取训练epoch数据
    hist['epoch'] = history.epoch
    # 打开绘图区
    plt.figure()
    # x轴标签
    plt.xlabel("训练次数", fontproperties=font)
    plt.ylabel("损失值", fontproperties=font)
    plt.plot(hist["epoch"], hist["mse"], label="Loss")
    # 打开图例
    plt.legend(prop=font)
    plt.savefig("./images/high-loss.png", format="png", dpi=300)
    # plt.plot(hist["epoch"], hist["val_mse"], label="Val Error")
    plt.show()


def train_model(model, inputs, outputs, model_path, log_path):
    ckpt_callback = callback_only_params(model_path)
    # tensorboard回调
    tensorboard_callback = tb_callback(log_path)
    # 保存参数
    model.save_weights(model_path.format(epoch=0))
    # 训练模型，并使用最新模型参数
    history = model.fit(
        inputs,
        outputs,
        epochs=300,
        callbacks=[ckpt_callback, tensorboard_callback],
        verbose=0
    )
    # 绘制图像
    plot_history(history)


def train_model_global(model, inputs, outputs, model_path):
    # 训练模型，并使用最新模型参数
    history = model.fit(
        inputs,
        outputs,
        epochs=300,
        verbose=1
    )
    model.save(model_path)


def load_model(model, model_path):
    # 检查最新模型
    latest = tf.train.latest_checkpoint(model_path)
    print("latest:{}".format(latest))
    # 载入模型
    model.load_weights(latest)


def prediction(model, model_path, inputs):
    # 载入模型
    load_model(model, model_path)
    # 预测值
    pres = model.predict(inputs)
    # print("prediction:{}".format(pres))
    # 返回预测值
    return pres


def plot_prediction(model, model_path, inputs, outputs):
    """可视化预测结果
    参数:
        model: 神经网络实例
        inputs: 输入数据
        outputs: 输出数据
        model_path: 模型文件路径
    返回:
        无
    """
    # 预测值
    pres = prediction(model, model_path, inputs)
    # 绘制理论值散点图
    plt.scatter(inputs, outputs, s=10, c="r", marker="*", label="实际值")
    # 绘制预测值曲线图
    plt.plot(inputs, pres, label="预测结果")
    plt.xlabel("输入数据", fontproperties=font)
    plt.ylabel("预测值", fontproperties=font)
    # 打开图例
    plt.legend(prop=font)
    # 保存绘制图像
    # plt.savefig("./images/line-fit-high.png", format="png", dpi=300)
    # 展示绘制图像
    plt.show()


def plot_prediction(model, model_path, inputs, outputs):
    # 预测值
    pres = prediction(model, model_path, inputs)
    # 绘制理论值散点图
    plt.scatter(inputs, outputs, s=10, c="r", marker="*", label="实际值")
    # 绘制预测值曲线图
    plt.plot(inputs, pres, label="预测结果")
    plt.xlabel("输入数据", fontproperties=font)
    plt.ylabel("预测值", fontproperties=font)
    # 打开图例
    plt.legend(prop=font)
    # 保存绘制图像
    # plt.savefig("./images/line-fit-high.png", format="png", dpi=300)
    # 展示绘制图像
    plt.show()


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    model_path = "./models/high-server/line-fit-high" + stamp
    log_path = "./logs/high/line-fit-high" + stamp
    inputs, outputs = gen_datas()
    model = create_model()
    # display_nn_structure(model, "./images/nn-structure-high.png")
    # train_model(model, inputs, outputs, model_path, log_path)
    model_path_global = "./models/high-global.h5"
    train_model_global(model, inputs, outputs, model_path_global)
    # model_path = "./models/high-server/"
    # plot_prediction(model, model_path, inputs, outputs)
