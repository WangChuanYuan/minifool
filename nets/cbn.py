from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model

"""
2 conv + pooling + BN
"""


class CBN(object):
    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', name='block1_conv1')(inputs)
        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', name='block1_conv2')(x)
        x = MaxPool2D(pool_size=(2, 2), name='block1_pool')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)
        x = Dropout(0.25)(x)

        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', name='fc')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(10, activation='softmax', name='softmax')(x)

        self.net = Model(inputs=[inputs], outputs=[predictions])

    def model(self):
        return self.net
