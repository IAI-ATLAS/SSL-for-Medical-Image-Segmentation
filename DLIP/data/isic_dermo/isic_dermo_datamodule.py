import os
import random

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.isic_dermo.isic_dermo_dataset import IsicDermoDataset

class IsicDermoDataModule(BasePLDataModule):
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
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        classifier_mode = False,
        classify_melanoma = False
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
        self.labeled_train_dataset: IsicDermoDataset = None
        self.unlabeled_train_dataset: IsicDermoDataset = None
        self.val_dataset: IsicDermoDataset = None
        self.test_dataset: IsicDermoDataset = None
        self.classifier_mode = classifier_mode
        self.classify_melanoma = classify_melanoma
        self.__init_datasets()
        self.assign_labeled_unlabeled_split()

    def __init_datasets(self):
        self.labeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms,
            classifier_mode = self.classifier_mode,
            classify_melanoma=self.classify_melanoma,
            balance_classes=True
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.unlabeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms_unlabeled,
            empty_dataset=True,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
            classifier_mode = self.classifier_mode,
            classify_melanoma=self.classify_melanoma
        )
        
        self.test_dataset = IsicDermoDataset(
            root_dir=self.test_root_dir,
            transforms=self.test_transforms,
            classifier_mode = self.classifier_mode,
            classify_melanoma=self.classify_melanoma
        )

        self.val_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.val_transforms,
            empty_dataset=True,
            classifier_mode = self.classifier_mode,
            classify_melanoma=self.classify_melanoma
        )
