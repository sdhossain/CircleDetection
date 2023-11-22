#-------------------------------------------------------------------#
# This is a script to train models and/or perform cross-validation  #
#                                                                   #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)                      #
#-------------------------------------------------------------------#

import os
import wandb
import yaml
import pandas as pd
from datetime import datetime
from wandb.keras import WandbCallback

from networks.architectures import regression_model
from dataset.dataloader import CircleDataLoader
from utils.misc import save_model_and_preds


cfg = yaml.full_load(open(os.path.join(os.getcwd(), 'config.yml'), 'r'))


def train_model(
    train_csv: str,
    val_csv: str,
    test_csv: str
) -> pd.DataFrame:
    """
    Trains tensorflow model for Circle Detection, Logs data to WandB
    and any appropriate logs. It makes a dataframe of predictions.

    Args:
        train_csv: path to train csv
        val_csv: path to csv with validation data
        test_csv: path to csv with test data
    
    Returns:
        pred_df: dataframe of test csv with predictions
    """

    wandb.init(job_type=cfg['TRAIN']['MODE'],
               project=cfg['WANDB']['PROJECT'],
               entity=cfg['WANDB']['ENTITY'])

    model, preprocess_fn = regression_model(
        input_shape=cfg['TRAIN']['INPUT_SHAPE'],
        opt=cfg['TRAIN']['OPTIMIZER'], 
        fc_layers=cfg['TRAIN']['ARCHITECTURE']['FC_LAYERS'],
        lr=cfg['TRAIN']['LEARNING_RATE'], 
        dropout=cfg['TRAIN']['ARCHITECTURE']['DROPOUT'], 
        l2_reg=cfg['TRAIN']['ARCHITECTURE']['L2_REG'], 
        l2_base=cfg['TRAIN']['ARCHITECTURE']['L2_BASE'], 
        base_architecture=cfg['TRAIN']['ARCHITECTURE']['BACKBONE'], 
        weights=cfg['TRAIN']['ARCHITECTURE']['WEIGHTS'],
    )

    dataloader = CircleDataLoader(
        dimensions=(cfg['TRAIN']['INPUT_SHAPE'][0],
                    cfg['TRAIN']['INPUT_SHAPE'][0]),
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        contrast_range=cfg['TRAIN']['CONTRAST_RANGE'],
        scale_fn=preprocess_fn
    )

    train_ds = dataloader.get_dataloader(csv_path=train_csv,
                                         shuffle=True,
                                         augment=True)
    val_ds = dataloader.get_dataloader(csv_path=val_csv,
                                       shuffle=False,
                                       augment=False)
    test_ds = dataloader.get_dataloader(csv_path=test_csv,
                                        shuffle=False,
                                        augment=False)
    model.fit(
        train_ds,
        epochs=cfg['TRAIN']['EPOCHS'],
        validation_data=val_ds,
        callbacks=[WandbCallback(save_model=False)],
        )

    pred_df = save_model_and_preds(
        model=model,
        preds_dir=cfg['TRAIN']['PREDICTIONS_DIR'],
        checkpoints_dir=cfg['TRAIN']['PREDICTIONS_DIR'],
        test_csv=test_csv,
        test_ds=test_ds,
        backbone=cfg['TRAIN']['ARCHITECTURE']['BACKBONE']
    )

    return pred_df


def kfold_cross_val() -> None:
    """
    Runs a k-fold cross validation
    """

    k = cfg['TRAIN']['KFOLD']
    full_results = pd.DataFrame()

    for fold in range(k):
        print(f"Training on fold {fold+1}/{k}...")

        train_csv = os.path.join(
            cfg['DATA']['ROOT_DIR'], f'splits/folds/train/{fold}.csv')
        val_csv = os.path.join(
            cfg['DATA']['ROOT_DIR'], f'splits/folds/val/{fold}.csv')

        pred_df = train_model(train_csv, val_csv, val_csv)
        pred_df['fold'] = fold

        full_results = pd.concat([full_results, pred_df], ignore_index=True)

    save_path = os.path.join(
        cfg['TRAIN']['PREDICTIONS_DIR'], 'cross_val_results.csv')
    full_results.to_csv(save_path, index=False)
    print(f"Cross-validation results saved to {save_path}")

    return


if __name__ == "__main__":
    if cfg['TRAIN']['MODE'] == 'single':
        train_model(
            train_csv=os.path.join(
                cfg['DATA']['ROOT_DIR'], 'splits/train.csv'),
            val_csv=os.path.join(
                cfg['DATA']['ROOT_DIR'], 'splits/val.csv'),
            test_csv=os.path.join(
                cfg['DATA']['ROOT_DIR'], 'splits/test.csv'),
        )

    if cfg['TRAIN']['MODE'] == 'cross-val':
        kfold_cross_val()

