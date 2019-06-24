import numpy as np
from keras import backend as K


class MIFGSMAttacker(object):
    def __init__(self, model, imgs, dimensions=(28, 28, 1)):
        self.model = model
        self.imgs = imgs
        self.dimensions = dimensions

    def attack_all(self, target=None, verbose=True):
        model_input_layer = self.model.layers[0].input
        model_output_layer = self.model.layers[-1].output

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

            # Create a copy of the input image to hack on
            hacked_image = np.copy(original_image)

            # Decide object_type_to_fake on whether is target attack
            object_type_to_fake = np.argmax(self.model.predict(original_image)[0]) if target is None else target

            # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
            cost_function = model_output_layer[0, object_type_to_fake]

            # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
            # In this case, referring to "model_input_layer" will give us back image we are hacking.
            gradient_function = K.gradients(cost_function, model_input_layer)[0]

            # Create a Keras function that we can call to calculate the current cost and gradient
            grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                            [cost_function, gradient_function])

            learning_rate = 0.0005
            momentum = 0
            decay_factor = 1

            itr = 0
            finish = False
            success = False
            while not finish:
                itr += 1

                # Check how close the image is to our target class and grab the gradients we
                # can use to push it one more step in that direction.
                # Note: It's really important to pass in '0' for the Keras learning mode here!
                # Keras layers behave differently in prediction vs. train modes!
                cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

                if target is None:
                    gradients = -gradients

                velocity = gradients / np.linalg.norm(np.reshape(gradients, self.dimensions[0] * self.dimensions[1]), ord=1)
                momentum = decay_factor * momentum + velocity
                hacked_image += np.sign(gradients) * learning_rate

                # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
                hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
                hacked_image = np.clip(hacked_image, 0.0, 1.0)

                if target is None:
                    if itr > 1500 or cost < 0.1:
                        finish = True
                        success = True if cost < 0.1 else False
                else:
                    if itr > 1500 or cost > 0.8:
                        finish = True
                        success = True if cost > 0.8 else False

                if verbose:
                    print("Iteration: {} Cost: {:.8}%".format(itr, cost * 100))
                    if finish:
                        print("Image{} attack success: {}".format(idx, success))
                        print("--------------------")  # end while

            hacked_image = hacked_image[0]
            hacked_image *= 255
            hacked_image = hacked_image.astype('uint8')

            hacked_images.append(hacked_image)

        return np.array(hacked_images)
