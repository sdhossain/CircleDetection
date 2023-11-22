import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from typing import Tuple, Callable
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils import normalize
from tensorflow.keras.models import load_model


def load_model_for_gradcam(
        model_path: str
    ) -> Tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """
    Loads in a model from SavedModel Tensorflow Checkpoints
    for the purpose of obtaining GradCAMs

    Args:
        model_path: path to saved checkpoints/saved model

    Returns:
        model, layer: a tuple of the model used for gradcam inference
            and the last convolutional layer
    """

    model = load_model(model_path)
    # The model's last convolutional layer is used for Grad-CAM
    target_layer = [
        layer for layer in model.layers if 'conv' in layer.name][-1]

    return model, target_layer


def compute_gradcam(
        model: tf.keras.Model,
        target_layer: tf.keras.layers.Layer,
        img: np.array,
        preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
    ) -> np.array:
    """
    Computes the Grad-CAM heatmap for a given image, focusing on
    the radius prediction. As GradCAM is mostly used for classification,
    we use the radius as a proxy for classification in an attempt to
    get some values - eventhough it is not a class-persay

    Args:
        model: The loaded TensorFlow model for which Grad-CAM
            is to be computed.
        target_layer: The specific layer in model for Grad-CAM.
        img: The input image for which the Grad-CAM heatmap is to be generated.
        preprocess_fn: The preprocessing function to apply to the image before feeding it to the model.

    Returns:
        heatmap: The computed Grad-CAM heatmap as a numpy array.
    """
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=False)
    processed_img = preprocess_fn(np.array([img]))

    # Generating heatmap - 
    score = CategoricalScore(2)
    cam = gradcam(score, processed_img, penultimate_layer=target_layer)
    heatmap = normalize(cam[0])
    return heatmap