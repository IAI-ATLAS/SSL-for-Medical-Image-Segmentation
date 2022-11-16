import os
import random
import logging

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.pneumonia_xray.pneumonia_xray_dataset import PnuemoniaXrayDataset

class PneumoniaXrayDataModule(BasePLDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        val_to_train_ratio,
        dataset_size = 1.0,
        initial_labeled_ratio=None,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=32,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        labels_available = True
    ):
        super().__init__(
            dataset_size,
            batch_size,
            val_to_train_ratio,
            initial_labeled_ratio,
            num_workers,
            pin_memory,
            shuffle,
            drop_last
        )
        self.val_to_train_ratio = val_to_train_ratio
        self.root_dir = root_dir
        self.labels_available = labels_available
        self.train_root_dir = os.path.join(self.root_dir, "train")
        self.test_root_dir = os.path.join(self.root_dir, "test")
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: PnuemoniaXrayDataset = None
        self.unlabeled_train_dataset: PnuemoniaXrayDataset = None
        self.val_dataset: PnuemoniaXrayDataset = None
        self.test_dataset: PnuemoniaXrayDataset = None
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = PnuemoniaXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms,
            labels_available=self.labels_available,
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.unlabeled_train_dataset = PnuemoniaXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms_unlabeled,
            empty_dataset=True,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
        )
        
        self.test_dataset = PnuemoniaXrayDataset(
            root_dir=self.test_root_dir,
            transforms=self.test_transforms,
            labels_available=self.labels_available,
        )

        self.val_dataset = PnuemoniaXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.val_transforms,
            empty_dataset=True,
            labels_available=self.labels_available,
        )

