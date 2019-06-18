from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, Dropout, Flatten

"""
2 conv + 3 fc
"""


class CFC(object):
    def __init__(self, input_shape):
        model = Sequential()

        model.add(BatchNormalization(input_shape=input_shape))

        model.add(
            Conv2D(64, kernel_size=(5, 5), activation='relu', bias_initializer='RandomNormal',
                   kernel_initializer='random_uniform', name='block1_conv1'))
        model.add(MaxPool2D(pool_size=(2, 2), name='block1_pool'))

        model.add(Conv2D(512, kernel_size=(5, 5), activation='relu', name='block2_conv1'))
        model.add(MaxPool2D(pool_size=(2, 2), name='block2_pool'))

        model.add(Flatten(name='flatten'))

        model.add(Dense(128, activation="relu", name='block3_fc1'))
        model.add(Dropout(0.35))
        model.add(Dense(64, activation='relu', name='block3_fc2'))
        model.add(Dropout(0.35))
        model.add(Dense(10, activation='softmax', name='softmax'))

        self.net = model

    def model(self):
        return self.net
