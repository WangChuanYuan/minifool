import numpy as np

from util.image_comparator import comparator
from util.differential_evolution import differential_evolution


class PixelAttacker(object):
    def __init__(self, model, imgs, dimensions=(28, 28, 1)):
        self.model = model
        self.imgs = imgs
        self.probs = model.predict(imgs / 255)
        self.dimensions = dimensions

    @staticmethod
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

    # The untargeted attack is to minimize the confidence of the model in the correct target class
    def predict_classes(self, xs, img, target_class, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = PixelAttacker.perturb_image(xs, img)
        predictions = self.model.predict(imgs_perturbed / 255)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def is_attack_success(self, x, img, target_class, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = PixelAttacker.perturb_image(x, img)

        confidence = self.model.predict(attack_image / 255)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or targeted classification), return True
        if (verbose):
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_idx, target=None, pixel_count=80,
               maxiter=15, popsize=400, verbose=True):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else np.argmax(self.probs[img_idx])

        dim_x, dim_y, dim_c = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        original_image = self.imgs[img_idx]

        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_classes(xs, original_image, target_class, target is None)
        callback_fn = lambda x, convergence: self.is_attack_success(x, original_image, target_class,
                                                                    targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        hacked_image = PixelAttacker.perturb_image(attack_result.x, original_image)[0]

        if verbose:
            prior_probs = self.model.predict(np.array([original_image]) / 255)[0]
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
