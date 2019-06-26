import numpy as np
from keras import Model
from keras.layers import Dense

from util.image_comparator import comparator
from util.differential_evolution import differential_evolution


class PixelAttacker(object):
    def __init__(self, model, imgs, dimensions=(28, 28, 1)):
        self.model = model
        self.imgs = imgs
        self.logits_model = PixelAttacker.get_logits_model(model)
        self.probs = model.predict(imgs / 255)
        self.dimensions = dimensions

    @staticmethod
    def perturb_image(xs, img):
        # If this function is passed just one perturbation vector,
        # pack it in a list to keep the computation the same
        if xs.ndim < 2:
            xs = np.array([xs])

        # create n new perturbed images
        tile = [len(xs)] + [1] * (xs.ndim + 1)
        imgs = np.tile(img, tile)

        xs = xs.astype(float)

        for x, img in zip(xs, imgs):
            # Split x into an array of 3-tuples (perturbation pixels)
            # i.e., [[x,y,c], ...]
            pixels = np.split(x, len(x) // 3)
            for pixel in pixels:
                x_pos, y_pos, c = pixel
                x_pos = int(x_pos)
                y_pos = int(y_pos)
                img[x_pos, y_pos, 0] = c

        return imgs

    @staticmethod
    def get_logits_model(model):
        dense_layer = Dense(10)
        logits = dense_layer(model.layers[-2].output)
        logits_model = Model(inputs=[model.layers[0].input], outputs=[logits])
        dense_layer.set_weights(model.layers[-1].get_weights())
        return logits_model

    def predict_logits(self, xs, img, target_class, targeted_attack=False):
        imgs_perturbed = PixelAttacker.perturb_image(xs, img)
        logits = self.logits_model.predict(imgs_perturbed)[:, target_class]
        return logits if not targeted_attack else -logits

    def is_attack_success(self, x, img, target_class, targeted_attack=False, verbose=False):
        images_perturbed = PixelAttacker.perturb_image(x, img)

        probs = self.model.predict(images_perturbed)[0]
        confidence = probs[target_class]

        if verbose:
            print('Confidence:', confidence)
        if ((targeted_attack and confidence > 0.9) or
                (not targeted_attack and confidence < 0.1)):
            return True

    def attack(self, img_idx, target=None, pixel_count=70,
               maxiter=15, popsize=400, verbose=True):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else np.argmax(self.probs[img_idx])

        dim_x, dim_y, dim_c = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 1)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        original_image = self.imgs[img_idx] / 255

        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_logits(xs, original_image, target_class, target is not None)
        callback_fn = lambda x, convergence: self.is_attack_success(x, original_image, target_class,
                                                                    targeted_attack, verbose)

        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        hacked_image = PixelAttacker.perturb_image(attack_result.x, original_image)[0]
        original_image *= 255
        original_image = original_image.astype('uint8')
        hacked_image *= 255
        hacked_image = hacked_image.astype('uint8')

        if verbose:
            prior_probs = self.probs[img_idx]
            predicted_probs = self.model.predict(np.array([hacked_image]) / 255)[0]
            predicted_class = np.argmax(predicted_probs)
            actual_class = np.argmax(prior_probs)
            is_success = predicted_class != actual_class
            cdiff = prior_probs[actual_class] - predicted_probs[actual_class]
            print("Image{} attack success:{}".format(img_idx, is_success))
            print("The predicted class is {}, the actual class is {}".format(actual_class, predicted_class))
            print("The difference between prior_prob and predicted_prob is {}".format(cdiff))
            print("SSIM: {}".format(comparator.compare(original_image, hacked_image, show=True)))
            print("--------------------")

        return hacked_image

    def attack_all(self, verbose=True):
        hacked_images = []
        for idx in range(len(self.imgs)):
            hacked_images.append(self.attack(idx, verbose=verbose))
        return np.array(hacked_images)
