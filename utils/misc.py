#------------------------------------------------------------#
# Miscellaneous utilities                                    #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

import os
import pandas as pd
import tensorflow as tf
from datetime import datetime


def save_model_and_preds(
        model: tf.keras.Model,
        preds_dir: str,
        checkpoints_dir: str,
        test_csv: str,
        test_ds: tf.data.Dataset,
        backbone: str
    ) -> pd.DataFrame:
    """
    Saves model predictions to a csv file and tensorflow model weights
    to a SavedModel format.

    Args:
        model: tensorflow model object (trained)
        preds_dir: directory where prediction csvs are saved
        checkpoints_dir: directory where checkpoints are saved
        test_csv: path to csv with test filepaths
        test_ds: tensorflow dataset of test data
        backbone: the convolutional backbone we use

    Returns:
        pred_df: dataframe of test csv with predictions
    """

    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Obtain predictions
    preds = model.predict(test_ds)
    test_df = pd.read_csv(test_csv)
    predictions_df = pd.DataFrame(
        preds, columns=['predicted_row', 'predicted_col', 'predicted_radius']
    )
    pred_df = pd.concat(
        [test_df.reset_index(drop=True), predictions_df], axis=1
    )
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model and predictions
    pred_df.to_csv(
        os.path.join(preds_dir, f"{cur_time}_{backbone}.csv")
    )
    model.save(
        os.path.join(checkpoints_dir, f"{cur_time}_{backbone}")
    )

    return pred_df
