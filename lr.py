#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn import datasets
from sklearn.utils import shuffle
from func import *
import matplotlib.pyplot as plt
from keras.datasets import imdb, mnist


# class LR(object):
#     def __init__(self, batch_size=128, epochs=3):
#         self.batch_size = batch_size
#         self.epochs = epochs
#         pass
#     def train(self):
#         pass
#     def query(self):
#         pass


batch_size = 128
epochs = 40
# 生成特征数据，符合正态分布，且增加噪声数据
# num_samples = 100000
# n_features = 3
# w_real = [2, -3, 1.5]
# b_real = 5
# x_train = tf.random.Generator.from_seed(1).normal((num_samples, n_features), stddev=1)
# y_train = x_train[:, 0]*w_real[0] + x_train[:, 1]*w_real[1] + x_train[:, 2]*w_real[2] + b_real
# y_train += tf.random.Generator.from_seed(1).normal(y_train.shape, stddev=0.01)
# y_train = tf.where(tf.sigmoid(y_train) > 0.5, 1, 0)

# 数据加载和处理
# imdb数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train[0])
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(x_train[0]))
x_train = keras.preprocessing.sequence.pad_sequences(x_train, truncating='post', padding='post', maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, truncating='post', padding='post', maxlen=256)
# x_train = tf.ragged.constant(x_train)
# x_test = tf.ragged.constant(x_test)
# train_data = tf.data.Dataset.from_tensor_slices((features, labels))
# train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# mnist数据
# n_features = 28*28
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255, x_test / 255
# x_train = x_train.reshape(-1, n_features).astype("float32") / 255
# x_test = x_test.reshape(-1, n_features).astype("float32") / 255
# y_train = y_train.astype("float32")
# y_test = y_test.astype("float32")

# iris数据
# n_features = 4
# data = datasets.load_iris()
# x_train, y_train = data.data, data.target.astype('float64')
# 两种shuffle方法，任选其一
# x_train, y_train = shuffle(x_train, y_train, random_state = 0)
# shuffle_in_unison_scary(x_train, y_train)
# 模型
# 方式一
# inputs = keras.Input(shape=(784,), name="digits")
# x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x = layers.Dense(64, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# 方式二
model = keras.Sequential()
model.add(layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.add(layers.Flatten(input_shape=(28, 28)))
# model.add(layers.Dense(10, activation='softmax'))
# model.add(layers.Dense(1, kernel_initializer='ones', input_shape=(3,)))
model.summary()
#优化器、损失函数和评估标准
model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['acc'])
# model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['mse'])
# 模型训练、评估、预测
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
results = model.evaluate(x_test, y_test, batch_size=batch_size)
pre = model.predict(x_test[:3])
print(pre)
print(y_test[:3])
history_dict = history.history
print(history_dict.keys())


# 画图
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
# “bo”代表 "蓝点"
# plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, acc, 'bo', label='Training acc')
# b代表“蓝色实线”
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# pd.DataFrame(history.history).plot(figsize=(20, 5))
# plt.grid(True)
# plt.xlabel('epoch')
# plt.show()

# 输出参数
# w, b = model.layers[0].get_weights()
# print(w,b)

# 存储模型
# model.save('saved_models/my_lr')
# model_variables = model.variables
# for v in model_variables:
#     print(v.name)
#     print(v.value)
