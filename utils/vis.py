#------------------------------------------------------------#
# This is a script to define utilities for visualizations    #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

import os
import pandas as pd
import numpy as np
from skimage import io
from typing import List, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
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
                                                img=img,
                                                backone=backbone,
                                                class_idx=class_idx,
                                                dimensions=dimensions)

            # GT and Predicted Circles
            plt.figure()
            ax = plt.gca()
            draw_circle(img=img, 
                        row=row['row'],
                        col=row['col'],
                        radius=row['radius'],
                        color='green',
                        ax=ax)
            draw_circle(img=img,
                        row=row['predicted_row'],
                        row=row['predicted_col'],
                        row=row['predicted_radius'],
                        color='red',
                        ax=ax)
            plt.legend(['Ground Truth', 'Prediction'])
            plt.title(
                f"Radius Range {min_radius}-{max_radius}," + \
                f"Size {row['radius']}"
            )
            plt.savefig(
                os.path.join(
                    directory,
                    f"radius_range_{min_radius}_{max_radius}_size" + \
                    f"_{row['radius']}_index_{index}.png")
            )
            plt.close()

            # Saliency Maps
            plt.figure()
            ax = plt.gca()
            ax.imshow(img, cmap='gray')
            ax.imshow(saliency_map, cmap='jet', alpha=0.5)
            plt.title(
                f"Saliency Map - Radius Range {min_radius}-" + \
                    f"{max_radius}, Size {row['radius']}")
            plt.savefig(
                os.path.join(
                    directory,
                    f"saliency_map_radius_range_{min_radius}_" + \
                        f"{max_radius}_size_{row['radius']}_index_{index}.png")
            )
            plt.close()
    
    return


def draw_circle(
        img: np.ndarray,
        row: int,
        col: int,
        radius: int,
        color: str = 'red',
        ax:plt.Axes = None
    ) -> None:
    """
    Function to draw a circle on the image.

    Args:
        img: Image on which we draw (unused)
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
    print(":3")

    return