import numpy as np


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 3-tuples (perturbation pixels)
        # i.e., [[x,y,c], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its gray value
            x_pos, y_pos, c = pixel
            img[x_pos, y_pos, 0] = c

    return imgs
