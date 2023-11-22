#--------------------------------------------------------------------#
# This is a script for explainability methods like saliency mapping  #
#                                                                    #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)                       #
#--------------------------------------------------------------------#

import numpy as np
import tensorflow as tf
from typing import Tuple
from src.networks.architectures import ARCHITECTURE_DICT


def compute_saliency_map(
        model: tf.keras.Model,
        img_path: str,
        backbone: str,
        class_idx: int,
        dimensions: Tuple[int, int],
    ) -> np.array:
    """
    Computes the saliency map for a given image.

    Args:
        model: The loaded TensorFlow model for computations
        img_path: Path to input image upon which we compute saliency map
        backbone: name of the architecture we are running explainability for
        class_idx: The index of the output neuron to focus on. In a regression
            model, this corresponds to a particular output feature.
            0: row, 1: col, 2: radius
        dimensions: (height, width) of desired images

    Returns:
        saliency_map: The computed saliency map as a numpy array.
    """

    preprocess_fn = ARCHITECTURE_DICT[backbone]['preprocess_fn']
    
    image_str = tf.io.read_file(img_path)
    image_decoded = tf.image.decode_png(image_str, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, dimensions)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.expand_dims(image, 0)

    if preprocess_fn is not None:
        img_processed = preprocess_fn(image)
    else:
        img_processed = image/ 255.0

    with tf.GradientTape() as tape:
        tape.watch(img_processed)
        predictions = model(img_processed)
        target_output = predictions[0][class_idx]

    grads = tape.gradient(target_output, img_processed)
    saliency_map = np.maximum(grads, 0)
    saliency_map = saliency_map / saliency_map.max()

    return saliency_map[0]