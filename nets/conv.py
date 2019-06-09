from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

"""
2 conv
"""


class Conv(object):
    def __init__(self, input_shape):
        model = Sequential()

        model.add(
            Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', name='block1_conv1'))
        model.add(MaxPool2D(pool_size=(2, 2), name='block1_pool'))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='block2_conv1'))
        model.add(MaxPool2D(pool_size=(2, 2), name='block2_pool'))

        model.add(Dropout(0.25))
        model.add(Flatten(name='flatten'))

        model.add(Dense(output_dim=128, activation="relu", name='fc'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=10, activation='softmax', name='softmax'))

        self.net = model

    def model(self):
        return self.net
