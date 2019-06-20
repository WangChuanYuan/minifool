import numpy as np
from keras.datasets import fashion_mnist
from keras.models import load_model

from adv.one_pixel_attack import PixelAttacker


def aiTest(images: np.ndarray, shape: tuple):
    model = load_model('models/cbn.h5')
    attacker = PixelAttacker(model, images)
    return attacker.attack_all()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)

    x = np.array(x_train[100:300])
    y = y_train[100:300]
    # samples = aiTest(x, x.shape)
    # np.save('samples100_300.npy', samples)
    model = load_model('models/cbn.h5')
    correct = model.evaluate(x/255, y)
    fool = model.evaluate(np.load('samples100_300.npy')/255, y)
