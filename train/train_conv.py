import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist

from nets.conv import Conv

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]
    conv = Conv(input_shape).model()
    conv.summary()
    conv.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    conv.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=64, epochs=50, verbose=1,
             callbacks=[ModelCheckpoint(filepath='../models/conv_v1.h5', save_best_only=True, monitor='val_acc')])
    score = conv.evaluate(x_test, y_test, verbose=1)
