import numpy as np
from keras.datasets import fashion_mnist
from keras.models import load_model

from adv import fgsm


def aiTest(images: np.ndarray, shape: tuple):
    return fgsm.attack(images, shape)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)

    x = np.array(x_train[100:200])
    y = y_train[100:200]
    # samples = aiTest(x, x.shape)
    model = load_model('models/conv.h5')
    fool = model.evaluate(np.load('adv/samples100_199.npy')/255, y)
