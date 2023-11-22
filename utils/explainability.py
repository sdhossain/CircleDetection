#--------------------------------------------------------------------#
# This is a script for explainability methods like saliency mapping  #
#                                                                    #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)                       #
#--------------------------------------------------------------------#

import numpy as np
import tensorflow as tf
from typing import Tuple
from networks.architectures import ARCHITECTURE_DICT


def compute_saliency_map(
        model: tf.keras.Model,
        img: np.array,
        backbone: str,
        class_idx: int,
        dimensions: Tuple[int, int],
    ) -> np.array:
    """
    Computes the saliency map for a given image.

    Args:
        model: The loaded TensorFlow model for computations
        img: The input image for which the saliency map is to be generated.
        backbone: name of the architecture we are running explainability for
        class_idx: The index of the output neuron to focus on. In a regression
            model, this corresponds to a particular output feature.
            0: row, 1: col, 2: radius
        dimensions: (height, width) of desired images

    Returns:
        saliency_map: The computed saliency map as a numpy array.
    """

    _, preprocess_fn = ARCHITECTURE_DICT[backbone]

    img_resized = tf.image.resize(img, dimensions)
    img_resized = tf.cast(img_resized, tf.float32)

    if preprocess_fn is not None:
        img_processed = preprocess_fn(img_resized[np.newaxis, ...])
    else:
        img_processed = img_resized[np.newaxis, ...] / 255.0

    with tf.GradientTape() as tape:
        tape.watch(img_processed)
        predictions = model(img_processed)
        target_output = predictions[0][class_idx]

    grads = tape.gradient(target_output, img_processed)
    saliency_map = np.maximum(grads, 0)
    saliency_map = saliency_map / saliency_map.max()

    return saliency_map[0]