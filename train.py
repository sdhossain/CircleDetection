import os
import wandb
import yaml
import pandas as pd
from datetime import datetime
from wandb.keras import WandbCallback

from networks.architectures import regression_model
from dataset.dataloader import CircleDataLoader


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
        test_df: dataframe of test csv with predictions
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

    os.makedirs(cfg['TRAIN']['PREDICTIONS_DIR'], exist_ok=True)
    os.makedirs(cfg['TRAIN']['CHECKPOINTS_DIR'], exist_ok=True)
    preds = model.predict(test_ds)
    test_df = pd.read_csv(test_csv)
    predictions_df = pd.DataFrame(
        preds, columns=['predicted_row', 'predicted_col', 'predicted_radius'])
    pred_df = pd.concat(
        [test_df.reset_index(drop=True), predictions_df], axis=1)
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_df.to_csv(
        os.path.join(
            cfg['TRAIN']['PREDICTIONS_DIR'],
             f"{cur_time}_{cfg['TRAIN']['ARCHITECTURE']['BACKBONE']}.csv"
        )
    )

    return pred_df


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

