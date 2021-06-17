#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import kerastuner as kt
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
# MPG数据
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# 数据清洗
dataset.isna().sum()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
# 拆分训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
# 数据规范化
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# fashion mnist数据
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# imdb数据
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# print(x_train[0])
# word_index = imdb.get_word_index()
#
# word_index = {k:(v+3) for k,v in word_index.items()}
# word_index["<PAD>"] = 0
# word_index["<START>"] = 1
# word_index["<UNK>"] = 2  # unknown
# word_index["<UNUSED>"] = 3
#
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# def decode_review(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_review(x_train[0]))
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, truncating='post', padding='post', maxlen=256)
# x_test = keras.preprocessing.sequence.pad_sequences(x_test, truncating='post', padding='post', maxlen=256)
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

# 建模
def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model
model = build_model()
# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=epochs,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop])
#
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model
# 用keras-tuner调参
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

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
