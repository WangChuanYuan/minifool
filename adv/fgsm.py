import numpy as np
from keras import backend as K
from keras.engine.training import Model
from keras.models import load_model


def attack(images: np.ndarray, shape: tuple):
    assert images.shape == shape

    model: Model = load_model('models/cbn.h5')
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    hacked_images = []
    idx = 0
    for image in images:
        idx += 1
        print("Generate adversarial sample for image {}".format(idx))
        # Add a 4th dimension for batch size
        original_image = np.expand_dims(image, axis=0)
        original_image = original_image.astype('float32')
        original_image /= 255

        max_change_above = original_image + 0.05
        max_change_below = original_image - 0.05

        # Create a copy of the input image to hack on
        hacked_image = np.copy(original_image)

        object_type_to_fake = np.argmin(model.predict(original_image)[0])

        # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
        cost_function = model_output_layer[0, object_type_to_fake]

        # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
        # In this case, referring to "model_input_layer" will give us back image we are hacking.
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        # Create a Keras function that we can call to calculate the current cost and gradient
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                        [cost_function, gradient_function])

        learning_rate = 0.005

        cost = 0.0
        itr = 0
        while cost < 0.8:
            # Check how close the image is to our target class and grab the gradients we
            # can use to push it one more step in that direction.
            # Note: It's really important to pass in '0' for the Keras learning mode here!
            # Keras layers behave differently in prediction vs. train modes!
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

            # Move the hacked image one step further towards fooling the model
            hacked_image += np.sign(gradients) * learning_rate

            # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
            hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
            hacked_image = np.clip(hacked_image, 0.0, 1.0)

            itr += 1
            print("Iteration: {} Cost: {:.8}%".format(itr, cost * 100))
            if itr > 1200:
                break

        hacked_image = hacked_image[0]
        hacked_image *= 255
        hacked_image = hacked_image.astype('uint8')

        hacked_images.append(hacked_image)

    return np.array(hacked_images)
