import numpy as np
from keras.datasets import fashion_mnist
from keras.models import load_model

from adv.fgsm import MIFGSMAttacker
from adv.one_pixel_attack import PixelAttacker


def aiTest(images: np.ndarray, shape: tuple):
    assert images.shape == shape

    target = load_model('models/cfc.h5')
    attacker = PixelAttacker(target, images)
    return attacker.attack_all(verbose=True)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)

    x = np.array(x_train[340:360]).astype('float32')
    y = y_train[340:360]
    samples = aiTest(x, x.shape)
    np.save('gen/samples340_360.npy', samples)
    model = load_model('models/cbn.h5')
    correct = model.evaluate(x/255, y)
    gen = np.load('gen/samples340_360.npy').astype('float32')
    fool = model.evaluate(gen/255, y)
