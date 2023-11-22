#------------------------------------------------------------#
# This is a script to perform evaluations                    #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

import os
import yaml
import pandas as pd
from datetime import datetime
from prettytable import PrettyTable

from utils.starter import CircleParams, iou
from utils.vis import plot_and_save_circles


THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
RADIUS_RANGES = [(5, 10), (10, 25), (25, 50), (50, 100)]

def calculate_accuracy(
        df: pd.DataFrame, threshold: float
    ) -> float:
    """
    Calculates accuracy given a threshold and dataframe of predicted
    circle parameters.

    Args:
        df: dataframe with ground truth and predicted circle parameters
        threshold: the IoU threshold upon which to inspect detections
    Returns:
        accuracy: the accuracy based upon the  defined threshold
    """

    ious = df.apply(
        lambda x: iou(CircleParams(row=x['row'],
                                   col=x['col'],
                                   radius=x['radius']), 
                      CircleParams(row=x['predicted_row'],
                                   col=x['predicted_col'],
                                   radius=x['predicted_radius'])),
                      axis=1)

    return (ious >= threshold).mean()


def evaluate_accuracy(df: pd.DataFrame) -> None:
    """
    Calculates accuracies by IoU threshold and report results for different
    sizes of radii to understand model performance.

    Args:
        df: dataframe with ground truth and predicted circle parameters
    Returns:
        does not return anything, prints out results of evaluation
    """

    thresholds = THRESHOLDS
    radius_ranges = RADIUS_RANGES  + [('All', None)]

    table = PrettyTable()
    table.field_names = ["Radius Range/IoU"] + [
        f"IoU ≥ {thres}" for thres in thresholds]

    for min_radius, max_radius in radius_ranges:
        radius_range = f"{min_radius}-{max_radius}" \
            if max_radius else 'All'
        row = [radius_range]

        for threshold in thresholds:
            radius_df = df[
                (df['radius'] >= min_radius) 
                & (df['radius'] < max_radius)] if max_radius else df

            if len(radius_df) > 0: 
                accuracy = calculate_accuracy(radius_df, threshold)
                row.append(f"{accuracy * 100:.2f}%")
            else:
                row.append(f"N/A")

        table.add_row(row)

    print(table)

    return

if __name__ == "__main__":
    cfg = yaml.full_load(
        open(os.path.join(os.getcwd(), 'config.yml'), 'r'))
    df = pd.read_csv(cfg['EVAL']['PREDICTIONS_CSV'])

    # Evaluate and final model fail/success cases
    evaluate_accuracy(df)

    # Visualize model performance and explainability
    plot_and_save_circles(df=df,
                          directory=os.path.join(
                              cfg['EVAL']['VIS_DIR'],
                              datetime.now().strftime("%Y%m%d_%H%M%S")),
                          radius_ranges=RADIUS_RANGES,
                          model_path=cfg['EVAL']['CHECKPOINT'],
                          backbone=cfg['EVAL']['BACKBONE'],
                          class_idx=cfg['EVAL']['SALIENCY_FOCUS'],
                          dimensions=cfg['EVAL']['INPUT_DIMS'])