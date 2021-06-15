#!/usr/bin/python
# -*- coding: utf-8 -*-
from tensorflow import keras

def main():
    print('hi')
    model = keras.models.load_model('saved_models/my_lr')
    w, b = model.layers[0].get_weights()
    print(w, b)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
