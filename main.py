import numpy as np
from keras.datasets import fashion_mnist
from keras.models import load_model

from adv.fgsm import MIFGSMAttacker


def aiTest(images: np.ndarray, shape: tuple):
    assert images.shape == shape

    target = load_model('models/cfc_v1.h5')
    attacker = MIFGSMAttacker(target, images)
    return attacker.attack_all(verbose=True)


if __name__ == '__main__':
    pass
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # x_train, x_test, y_train, y_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1), np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)
    #
    # start = 1000
    # end = 2000
    # x = np.array(x_test[start:end]).astype('float32')
    # y = y_test[start:end]
    # samples = aiTest(x, x.shape)
    # np.save('gen/samples' + str(start) + '_' + str(end) + '.npy', samples)
    # model = load_model('models/cbn.h5')
    # correct = model.evaluate(x/255, y)
    # gen = np.load('gen/samples' + str(start) + '_' + str(end) + '.npy').astype('float32')
    # fool = model.evaluate(gen/255, y)
