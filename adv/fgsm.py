from collections import Iterable

import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import Model


class MIFGSMAttacker(object):
    def __init__(self, model, imgs, dimensions=(28, 28, 1)):
        self.model = model
        self.imgs = imgs
        self.dimensions = dimensions

    def get_logits_model(self):
        dense_layer = Dense(10)
        logits = dense_layer(self.model.layers[-2].output)
        logits_model = Model(inputs=[self.model.layers[0].input], outputs=[logits])
        dense_layer.set_weights(self.model.layers[-1].get_weights())
        return logits_model

    def attack_all(self,
                   epsilons=0.0005,
                   epsilons_max=0.005,
                   steps=100,
                   epsilon_steps=100,
                   decay_factor=1,
                   target=None,
                   verbose=True):
        logits_model = self.get_logits_model()
        model_input = logits_model.layers[0].input
        model_output = logits_model.layers[-1].output

        hacked_images = []
        idx = 0
        for image in self.imgs:
            idx += 1
            # Add a 4th dimension for batch size
            original_image = np.expand_dims(image, axis=0)
            original_image = original_image.astype('float32')
            original_image /= 255

            max_change_above = original_image + 0.05
            max_change_below = original_image - 0.05

            hacked_image = np.copy(original_image)

            # Decide target_class on whether is target attack
            actual_class = np.argmax(self.model.predict(original_image)[0])
            target_class = actual_class if target is None else target

            # Create a Keras function
            cost_fn = model_output[0, target_class]
            predict_fn = self.model.layers[-1].output[0]
            gradient_fn = K.gradients(cost_fn, model_input)[0]

            # Note: It's really important to pass in '0' for the Keras learning mode when used
            cost_and_gradients_from_logits_model = K.function([model_input, K.learning_phase()], [cost_fn, gradient_fn])
            predict_from_model = K.function([model_input, K.learning_phase()], [predict_fn])

            if not isinstance(epsilons, Iterable):
                epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

            finish = False
            success = False
            step = 0
            for epsilon in epsilons[:]:
                if success or finish:
                    break
                if epsilon == 0.0:
                    continue
                momentum = 0
                for i in range(steps):
                    step += 1

                    cost, gradients = cost_and_gradients_from_logits_model([hacked_image, 0])

                    if target is None:
                        gradients = -gradients

                    velocity = gradients / np.mean(np.abs(gradients), axis=(1, 2, 3), keepdims=True)
                    momentum = decay_factor * momentum + velocity
                    hacked_image += np.sign(gradients) * epsilon

                    # Ensure that the image doesn't change too much
                    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
                    hacked_image = np.clip(hacked_image, 0.0, 1.0)

                    probs = predict_from_model([hacked_image, 0])[0]
                    predict_class = np.argmax(probs)
                    confidence = probs[target_class]

                    if verbose:
                        print("Step{}: confidence {:.8}%".format(step, confidence * 100))

                    if step > 600:
                        finish = True
                        break

                    if target is None:
                        if confidence < 0.05 and target_class != predict_class:
                            success = True
                            break
                    else:
                        if confidence > 0.9 and target_class == predict_class:
                            success = True
                            break  # End inner for
            # End outer for

            hacked_image = hacked_image[0]
            hacked_image *= 255
            hacked_image = hacked_image.astype('uint8')
            hacked_images.append(hacked_image)

            print("Image{} attack success: {}".format(idx, success))
            print("--------------------")

        return np.array(hacked_images)
