#------------------------------------------------------------#
# This is a script to get dataloaders for tensorflow         #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable
from tensorflow.keras.layers import RandomContrast


class CircleDataLoader:
    """
    Creates a Circle-TF dataset from image filenames and corresponding labels.
    Includes data preprocessing functionality, including resizing and
    data augmentation.
    """

    def __init__(
            self, 
            dimensions: Tuple[int, int],
            batch_size: int,
            contrast_range: float = 0.15,
            scale_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
        ) -> None:
        """
        Constructor of CircleDataLoader Object/Class

        Args:
            dimensions: (height, width) of desired images
            batch_size: Desired batch size
            contrast_range: (deviation) bounds for random contrast
                data augmentation.
            scale_fn: Model-specific preprocessing function
        """

        self.dims = dimensions
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE
        self.input_scaler = scale_fn
        self.tf_aug = tf.keras.Sequential([
            RandomContrast(contrast_range)
            ])

        return


    def get_dataloader(
            self,
            csv_path: str,
            shuffle: bool = False,
            augment: bool = False
        ) -> tf.data.Dataset:
        """
        Maps a series of preprocessing functions to each item in a dataset
        
        Args:
            csv_path: path to csv file containing file paths and labels
            augment: whether to use augmentations for the dataset
            shuffle: whether to shuffle the dataset

        Returns:
            dataloader: returns a tensorflow dataset
        """

        # Read in File Names and Labels
        df = pd.read_csv(csv_path)
        ds = tf.data.Dataset.from_tensor_slices(
            (df['filepath'].tolist(), np.array(
                df[['row', 'col', 'radius']].values.tolist(),
                dtype=np.float32))
        )

        if shuffle:
            ds = ds.shuffle(ds.cardinality())

        # Load, Resize, Augment and Batch data
        ds = ds.map(self._parse_fn, num_parallel_calls=self.autotune)
        ds = ds.map(
            lambda x, y: (
                self.tf_aug(x, training=True), y
                ),
            num_parallel_calls=self.autotune
        ) if augment else ds
        ds = ds.batch(self.batch_size)

        # Scale Input Data
        if self.input_scaler is None:
            ds = ds.map(
                lambda x, y: (x / 255., y),
                num_parallel_calls=self.autotune
            )
        else:
            ds = ds.map(
                lambda x, y: (self.input_scaler(x), y),
                num_parallel_calls=self.autotune
            )

        return ds.prefetch(buffer_size=self.autotune)

    def _parse_fn(
            self,
            image_filename: str,
            label: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Parse image and csv file info, resizes and extract labesl.
        Produces a tuple consisting of a resized image and label

        Args:
            image_filename (str): File name of the image
            label: Tensor of label

        Returns:
            (image, label): tuple consisting of the resized image and label
        """

        image_str = tf.io.read_file(image_filename)
        image_decoded = tf.image.decode_png(image_str, channels=1)
        image = tf.cast(image_decoded, tf.float32)

        return tf.image.grayscale_to_rgb(
            tf.image.resize(image, self.dims)), label
