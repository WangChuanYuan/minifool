import numpy as np

import util.perturber as pb
from util.differential_evolution import differential_evolution


class PixelAttacker(object):
    def __init__(self, model, imgs, dimensions=(28, 28, 1)):
        self.model = model
        self.imgs = imgs
        self.probs = model.predict(imgs / 255)
        self.dimensions = dimensions

    # The untargeted attack is to minimize the confidence of the model in the correct target class
    def predict_classes(self, xs, img, target_class, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = pb.perturb_image(xs, img)
        predictions = self.model.predict(imgs_perturbed / 255)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def is_attack_success(self, x, img, target_class, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = pb.perturb_image(x, img)

        confidence = self.model.predict(attack_image / 255)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or targeted classification), return True
        if (verbose):
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_idx, target=None, pixel_count=40,
               maxiter=10, popsize=400, verbose=True):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else np.argmax(self.probs[img_idx])

        dim_x, dim_y, dim_c = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_classes(xs, self.imgs[img_idx], target_class, target is None)
        callback_fn = lambda x, convergence: self.is_attack_success(x, self.imgs[img_idx], target_class,
                                                                    targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        hacked_image = pb.perturb_image(attack_result.x, self.imgs[img_idx])[0]
        prior_probs = self.model.predict(np.array([self.imgs[img_idx]]) / 255)[0]
        predicted_probs = self.model.predict(np.array([hacked_image]) / 255)[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = np.argmax(prior_probs)
        is_success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        if verbose:
            print("Image{} attack success:{}".format(img_idx, is_success))

        return hacked_image

    def attack_all(self):
        hacked_images = []
        for idx in range(len(self.imgs)):
            hacked_images.append(self.attack(idx))
        return hacked_images
