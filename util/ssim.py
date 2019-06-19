import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ssim


def cal_ssim(image1: np.ndarray, image2: np.ndarray):
    tf.reset_default_graph()
    with tf.Session() as sess:
        ssim_val = sess.run(ssim(tf.convert_to_tensor(image1), tf.convert_to_tensor(image2), 255))
        print("SSIM: {}".format(ssim_val))
