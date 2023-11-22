#------------------------------------------------------------#
# This is a script to define utilities for visualizations    #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

from typing import List, Tuple

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize
from utils.explainability import compute_saliency_map


def plot_and_save_circles(
        df: pd.DataFrame,
        directory: str,
        radius_ranges: List[Tuple[int]],
        model_path: str,
        backbone: str,
        class_idx: int,
        dimensions: Tuple[int, int],
    ) -> None:
    """
    Plots ground truth and predicted circles on images so that
    model actions can be inspected. Also plots and saves saliency maps.

    Args:
        df: dataframe with filepaths, ground truths and predictions
        directory: path/directory where visualized images are stored
        radius_ranges: ranges of radii to produce separate plots
        model: path to tensorflow SavedModel checkpoints
        backbone: name of the architecture we are running explainability for
        class_idx: The index of the output neuron to focus on. In a regression
            model, this corresponds to a particular output feature.
            0: row, 1: col, 2: radius
        dimensions: (height, width) of desired images
    
    Returns:
        does not return anything, only plots and save images
    """

    os.makedirs(directory, exist_ok=True)
    model = tf.keras.models.load_model(model_path)

    for min_radius, max_radius in radius_ranges:
        range_df = df[
            (df['radius'] >= min_radius) & (df['radius'] < max_radius)]
        examples = range_df.sample(n=min(3, len(range_df)))  # No. of Samples

        for index, row in examples.iterrows():
            img = io.imread(row['filepath'])
            saliency_map = compute_saliency_map(model=model,
                                                img_path=row['filepath'],
                                                backbone=backbone,
                                                class_idx=class_idx,
                                                dimensions=dimensions)

            # Original Image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Circles Overlays
            axes[1].imshow(img, cmap='gray')
            draw_circle(row['row'], row['col'], row['radius'],
                        color='green',ax=axes[1])
            draw_circle(row['predicted_row'], row['predicted_col'],
                        row['predicted_radius'], color='red', ax=axes[1])
            axes[1].set_title('Ground Truth and Predictions')
            axes[1].axis('off')

            # Saliency Mmap
            axes[2].imshow(img, cmap='gray')
            axes[2].imshow(
                resize(saliency_map, (img.shape[0], img.shape[1])),
                cmap='jet', alpha=0.6
            )
            axes[2].set_title('Saliency Map')
            axes[2].axis('off')

            plt.savefig(
                os.path.join(
                    directory,
                    f"combined_radius_range_{min_radius}" + \
                    f"_{max_radius}_size_{row['radius']}_" + \
                    f"index_{index}.png")
            )
            plt.close(fig)
    
    return


def draw_circle(
        row: int,
        col: int,
        radius: int,
        color: str = 'red',
        ax:plt.Axes = None
    ) -> None:
    """
    Function to draw a circle on the image.

    Args:
        row: row - co-ordinate of centre
        col: col - co-ordinate of centre
        radius: radius of circle
        color: color to draw circle
        ax: matplotlib ax object to draw on

    Returns:
        void function, but draws circle on figure
    """

    if ax is None:
        ax = plt.gca()
    circle = plt.Circle((col, row), radius, color=color, fill=False)
    ax.add_artist(circle)

    return