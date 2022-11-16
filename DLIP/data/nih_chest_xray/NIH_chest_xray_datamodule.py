import os
import random
import logging

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.nih_chest_xray.NIH_chest_xray_dataset import NIHChestXrayDataset

class NIHChestXrayDataModule(BasePLDataModule):
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
        self.labeled_train_dataset: NIHChestXrayDataset = None
        self.unlabeled_train_dataset: NIHChestXrayDataset = None
        self.val_dataset: NIHChestXrayDataset = None
        self.test_dataset: NIHChestXrayDataset = None
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = NIHChestXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms,
            labels_available=self.labels_available,
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.unlabeled_train_dataset = NIHChestXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms_unlabeled,
            empty_dataset=True,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
        )
        
        self.test_dataset = NIHChestXrayDataset(
            root_dir=self.test_root_dir,
            transforms=self.test_transforms,
            labels_available=self.labels_available,
        )

        self.val_dataset = NIHChestXrayDataset(
            root_dir=self.train_root_dir,
            transforms=self.val_transforms,
            empty_dataset=True,
            labels_available=self.labels_available,
        )
        
    def init_val_dataset(self, split_lst=None):
        if len(self.labeled_train_dataset)>0:
            # default init is random
            if split_lst is None:
                num_val_samples = int(round(len(self.labeled_train_dataset) * (self.val_to_train_ratio)))
                num_val_samples = num_val_samples if num_val_samples > 0 else 1
                added_samples = 0
                while added_samples < num_val_samples:
                    index = random.randrange(len(self.labeled_train_dataset))
                    # go to start of sequence
                    while int(self.labeled_train_dataset.indices[index].split('_')[1].split('.')[0]) != 0:
                        index-=1
                    indices = [index]
                    index+=1
                    while int(self.labeled_train_dataset.indices[index].split('_')[1].split('.')[0]) != 0:
                        indices.append(index)
                        index+=1
                    for indx in indices[::-1]:
                        self.val_dataset.add_sample(self.labeled_train_dataset.pop_sample(indx))
                        added_samples+=1
            else:
                ind_lst = []
                for elem in split_lst:
                    ind_lst.append(self.labeled_train_dataset.indices[elem])

                for ind_elem in ind_lst:
                    rem_ind = self.labeled_train_dataset.indices.index(ind_elem)
                    self.val_dataset.add_sample(
                        self.labeled_train_dataset.pop_sample(
                            rem_ind
                        )
                    )
        logging.warn("Init validation not possible due to no labeled training data.")
