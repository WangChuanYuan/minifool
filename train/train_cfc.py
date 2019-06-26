import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

from nets.cfc import CFC

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]
    cfc = CFC(input_shape).model()
    cfc.summary()
    cfc.compile(optimizer=Adam(decay=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])
    cfc.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=256, epochs=30, verbose=1,
             callbacks=[ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001),
                        ModelCheckpoint(filepath='../models/cfc_v1.h5', save_best_only=True, monitor='val_acc')])
    score = cfc.evaluate(x_test, y_test, verbose=1)
