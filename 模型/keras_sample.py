import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def line_fit_model():
    inputs = tf.keras.Input(shape=[1, ], name="inputs")
    layer1 = layers.Dense(10, activation='relu', name='layer1')(inputs)
    layer2 = layers.Dense(15, activation='relu', name='layer2')(layer1)
    outputs = layers.Dense(5, activation='softmax', name='outputs')(layer2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    keras.utils.plot_model(model, "F:/tensorflow/images/line_fit_model.png", show_shapes=True)
    return model


if __name__ == "__main__":

    line_fit_model()
