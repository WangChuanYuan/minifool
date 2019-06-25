import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ssim


class Comparator(object):
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            image1 = tf.placeholder(tf.uint8, shape=(28, 28, 1), name='image1')
            image2 = tf.placeholder(tf.uint8, shape=(28, 28, 1), name='image2')
            ssim_val = ssim(image1, image2, 255)
        self.graph = graph
        self.image1 = image1
        self.image2 = image2
        self.ssim = ssim_val

    def compare(self, image1, image2, show=False):
        if show:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.squeeze(image1), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(np.squeeze(image2), cmap='gray')
            plt.show()
        with tf.Session(graph=self.graph) as sess:
            ssim_val = sess.run(self.ssim, feed_dict={self.image1: image1, self.image2: image2})
        return ssim_val


comparator = Comparator()
