import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ssim


def show_img(image: np.ndarray):
    img = Image.fromarray(np.squeeze(image))
    img.show()


def cal_ssim(image1: np.ndarray, image2: np.ndarray):
    tf.reset_default_graph()
    with tf.Session() as sess:
        ssim_val = sess.run(ssim(tf.convert_to_tensor(image1), tf.convert_to_tensor(image2), 255))
        print("SSIM: {}".format(ssim_val))


def compare(image1, image2, show=True, ssim=True):
    if show:
        show_img(image1)
        show_img(image2)
    if ssim:
        cal_ssim(image2, image2)
