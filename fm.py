#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb, mnist


class FMLayer(keras.Model):
    def __init__(self, units, input_dim, embedding_size=32):
        super(FMLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        self.b = tf.Variable([0.0])
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(
        #     initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        # )
        v_init = tf.random_normal_initializer()
        self.v = tf.Variable(
            initial_value=v_init(shape=(input_dim, embedding_size), dtype="float32"),
            trainable=True,
        )
        # self.v = keras.layers.Embedding(batch_size, embedding_size, embeddings_initializer='uniform', input_length=input_dim)

    def call(self, inputs):
        bias = self.b
        first_order = tf.matmul(inputs, self.w)
        # v = tf.squeeze(self.v, 0)
        part1 = tf.matmul(inputs, self.v)
        second_order_part1 = tf.reduce_sum(tf.multiply(part1, part1))
        part2 = tf.multiply(tf.transpose(tf.square(self.v)), tf.square(inputs))
        second_order_part2 = tf.reduce_sum(part2)
        second_order = 0.5*tf.subtract(second_order_part1, second_order_part2)

        return bias+first_order+second_order


n_features = 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, n_features).astype("float32") / 255
x_test = x_test.reshape(-1, n_features).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# lr
# inputs = keras.Input(shape=(n_features,), name="digits")
# outputs = keras.layers.Dense(10, activation="softmax", name="predictions")(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs)

# fm
inputs = keras.Input(shape=(n_features,), name="digits")
x = FMLayer(10, n_features)(inputs)
outputs = keras.layers.Dense(10, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# 编译训练
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)