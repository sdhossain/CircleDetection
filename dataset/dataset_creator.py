#------------------------------------------------------------#
# This is a script to create a dataset of an arbitrary size  #
#                                                            #
# Author: Saad Hossain (s42hossa@uwaterloo.ca)               #
#------------------------------------------------------------#

import sys
sys.path.append('.')

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from utils.starter import generate_examples


class CircleDatasetCreator:

    def __init__(
        self,
        root_dir: str,
        exists_ok: bool = False,
        samples: int = 200000,
        train_prop: float = 0.8,
        val_prop: float = 0.1,
        test_prop: float = 0.1,
        folds: int = 10,
        img_size: int = 100,
        min_radius: int = 10,
        max_radius: int = 50,
        noise_level: int = 0.5
    ) -> None:
        """
        Constructor for CircleDatasetCreator. It allows building of a dataset
        from the generator provided in utils/starter.py and allows for
        reproducibility and inspection of data.

        Args:
            root_dir: directory where we would like to store the data
            exists_ok: whether to see if a current directory before generating
                one to avoid loss of data
            samples: number of total images/samples in our dataset
            train_prop: proportion of total data used for training
            val_prop: proportion of total data used for validation
            test_prop: proportion of total data used for testing
            folds: number of folds for K-Fold, cross validation
            img_size: height and width (in pixels) for images generated
            min_radius: minimum radius (in unit pixels) for circles
            max_radius: maximum radius (in unit pixels) for circles
            noise_level: standard deviation of normal distribution for noise
        Returns:
            dataset_creator object (constructor method)
        """

        assert sum([train_prop, val_prop, test_prop]) == 1, \
            "train, val and test proportions should sum to 1"

        self.root_dir = root_dir
        self.exists_ok = exists_ok
        self.samples = samples
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.folds = folds
        self.img_size = img_size
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.noise_level = noise_level

        return

    def generate_images_annotations(self) -> None:
        """
        Generates images with annotations pairing filepaths to annotations
        in a csv format.

        Results in the following directory structure:
        - root_dir
            - Images
            - annotations.csv
        """

        gen = generate_examples(
            img_size=self.img_size,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            noise_level=self.noise_level
        )

        images_dir = os.path.join(self.root_dir, "Images")
        os.makedirs(images_dir, exist_ok=self.exists_ok)

        annotations = []
        for i in range(self.samples):
            img, params = next(gen)
            img_path = os.path.join(images_dir, f"image_{i}.png")
            plt.imsave(img_path, img, cmap='gray')
            annotations.append(
                [img_path, params.row, params.col, params.radius]
            )

        df = pd.DataFrame(
            annotations, columns=['filepath', 'row', 'col', 'radius'])
        df.to_csv(
            os.path.join(self.root_dir, 'annotations.csv'), index=False)

        return

    def generate_splits(self) -> None:
        """
        Splits annotations into csv files for the appropriate use-case.

        Results in the following directory structure:
        - root_dir
            - Images
            - annotations.csv
            - splits
                - train.csv
                - val.csv
                - test.csv
                - folds
                    - train
                        - 0.csv
                        - 1.csv
                        ...
                    - val
                        - 0.csv
                        - 1.csv
                        ...
        """

        df = pd.read_csv(
            os.path.join(self.root_dir, 'annotations.csv'))
        val_size = self.val_prop / (1 - self.test_prop)

        # Train-Val-Test Splits
        train_df, test_df = train_test_split(
            df, test_size=self.test_prop, random_state=42)
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=42)

        splits_dir = os.path.join(self.root_dir, "splits")
        os.makedirs(splits_dir, exist_ok=self.exists_ok)

        train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

        # K-Fold Splits
        kf = KFold(n_splits=self.folds)
        fold_dir_train = os.path.join(splits_dir, "folds/train")
        fold_dir_val = os.path.join(splits_dir, "folds/val")
        os.makedirs(fold_dir_val, exist_ok=self.exists_ok)
        os.makedirs(fold_dir_train, exist_ok=self.exists_ok)

        for i, (train_index, val_index) in enumerate(kf.split(train_df)):
            train_fold_df = train_df.iloc[train_index]
            val_fold_df = train_df.iloc[val_index]
            train_fold_df.to_csv(os.path.join(fold_dir_train, f'{i}.csv'), index=False)
            val_fold_df.to_csv(os.path.join(fold_dir_val, f'{i}.csv'), index=False)

        return


    def build_dataset(self) -> None:
        """
        Builds dataset, such that it can be used for training/testing
        """

        self.generate_images_annotations()
        self.generate_splits()

        return


if __name__ == "__main__":

    cfg = yaml.full_load(
        open(os.path.join(os.getcwd(), 'config.yml'), 'r'))

    ds_creator = CircleDatasetCreator(
        root_dir=cfg['DATA']['ROOT_DIR'],
        exists_ok=cfg['DATA']['REPLACE'],
        samples=cfg['DATA']['SAMPLES'],
        train_prop=cfg['DATA']['TRAIN_PROPORTION'],
        val_prop=cfg['DATA']['VAL_PROPORTION'],
        test_prop=cfg['DATA']['TEST_PROPORTION'],
        folds=cfg['DATA']['FOLDS'],
        img_size=cfg['DATA']['IMG_SIZE'],
        min_radius=cfg['DATA']['MIN_RADIUS'],
        max_radius=cfg['DATA']['MAX_RADIUS'],
        noise_level=cfg['DATA']['NOISE_LEVEL']
    )
    ds_creator.build_dataset()

