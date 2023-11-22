#------------------------------------------------------------#
# This is a script to create convolutional neural networks   #
# for our regression task                                    #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

from typing import Optional, Tuple, Callable, Any, List

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, Input, \
                                    Activation, GlobalAveragePooling2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import Constant
from classification_models.keras import Classifiers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.applications import MobileNet, MobileNetV3Small


# Architecture dictionary to extract backbones with imagenet pre-trained
# weights for common architectures

ResNet18, resnet18_preprocess = Classifiers.get('resnet18')
ARCHITECTURE_DICT = {
    'resnet50': {
        'func': ResNet50,
        'preprocess_fn': tf.keras.applications.resnet50.preprocess_input
    }, 
    'resnet18': {
        'func': ResNet18,
        'preprocess_fn': resnet18_preprocess
    },
    'vgg16': {
        'func': VGG16,
        'preprocess_fn': tf.keras.applications.vgg16.preprocess_input
    },
    'mobilenet': {
        'func': MobileNet,
        'block_idxs': None,
        'preprocess_fn': None
    },
    'mobilenetv3': {
        'func': MobileNetV3Small,
        'preprocess_fn': tf.keras.applications.mobilenet_v3.preprocess_input
    },
}


def regression_model(
        input_shape: List[int] = [224, 224, 3], 
        metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        output_bias: Optional[Any] = None, 
        opt: str = 'adam', 
        fc_layers: List[int] = [32],
        lr: float = 0.01, 
        dropout: float = 0.01, 
        l2_reg: float = 0.001, 
        l2_base: float = 0, 
        base_architecture: str = 'mobilenetv2', 
        weights: str = 'imagenet',
    ) -> Tuple[tf.keras.Model, Callable[[tf.Tensor], tf.Tensor]]:
    '''
    Defines a model based on a pretrained from imagenet classifier
    (backbone used only and adapted for regression task)

    Args:
        input_shape: The shape of the model input
        metrics: Metrics to track model's performance
        output_bias: bias initializer of output layer (unused)
        opt: optimizer to use, one of adam or sgd
        fc_layers: list of nodes in fully connected layers after
            features pass through convolutional backbone
        lr: learning_rate of optimizer
        dropout: dropout parameter of fully connected layers
        l2_reg: l2 regularization penalty on fully connected layers
        l2_base: l2 regularization penalty for base model (conv-backbone)
        base_architecture: base model from keras.applications
        weights: str 'imagenet' or path to pre-trained weights

    Returns:
        (model, preprocessing_function): a Keras Model object with the
        architecture defined in this method + preprocessing function
    '''

    architecture_func = ARCHITECTURE_DICT[base_architecture]['func']

    if opt == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr)

    if output_bias is not None:
        output_bias = Constant(output_bias)     

    X_input = Input(input_shape, name='input', dtype='float32')
    base_model = architecture_func(include_top=False,
                                   weights=weights,
                                   input_shape=input_shape,
                                   input_tensor=X_input)

    if l2_base > 0:
        base_regularizer = L2(l2_base)
        for layer in base_model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, base_regularizer)
    
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)

    for nodes in fc_layers:
        X = Dense(nodes,
                  kernel_regularizer=L2(l2_reg),
                  bias_regularizer=L2(l2_reg),
                  activation='relu')(X)
        X = Dropout(dropout)(X)

    X = Dense(3,
              bias_initializer=output_bias,
              kernel_regularizer=L2(l2_reg),
              bias_regularizer=L2(l2_reg),
              name='logits')(X)
    Y = Activation(None,
                   dtype='float32',
                   name='output')(X)

    model = tf.keras.Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        metrics=metrics
    )

    return model, ARCHITECTURE_DICT[base_architecture]['preprocess_fn']