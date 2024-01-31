import numpy as np
from torch.utils.data import DataLoader
import json

from .base import BaseDictDataset, BaseDataModule
from .augmentations import aug


class SiameseEncodersDataset(BaseDictDataset):
    """Dataset for training siamese encoders. Samples two fragments from the same clip,
    performs augmentations and returns them as a dict"""

    def __init__(self, *args, **kwargs):
        super(SiameseEncodersDataset, self).__init__(*args, **kwargs)

        # Positive augmentations are already defined in base class

    def getitem(self, item, file=None, group_name=None):
        fragment1 = self.get_fragment(file)  # Gets first positive example
        fragment2 = self.get_fragment(file)  # Gets second positive example
        return fragment1, fragment2

    # Overriding augment to include second fragment
    def augment(self, fragment_tuple):
        fragment1, fragment2 = fragment_tuple
        override = self.augmentations_pos if self.transform_override else False

        fragment1 = aug(
            np.cast["float32"](fragment1),
            self.augmentations,
            override=override,
            sample_rate=self.sr,
        )

        fragment2 = aug(
            np.cast["float32"](fragment2),
            self.augmentations,
            override=override,
            sample_rate=self.sr,
        )

        return fragment1, fragment2

    def return_data(self, result=None, group_name=None, idx=None):
        fragment1, fragment2 = result
        return {"clip1": fragment1, "clip2": fragment2, "group_name": group_name}


class SiameseEncodersDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super(SiameseEncodersDataModule, self).__init__(**kwargs)
        self.batch_sampling_mode = self.dataset_kwargs["batch_sampling_mode"]
        self.sr = self.dataset_kwargs["sr"]
        self.nr_samples = self.dataset_kwargs["nr_samples"]
        self.multi_epoch = self.dataset_kwargs["multi_epoch"]
        self.normalize = self.dataset_kwargs["normalize"]
        self.transform_override = self.dataset_kwargs["transform_override"]
        self.augmentations = self.dataset_kwargs["augmentations"]

    def train_dataloader(self):
        return DataLoader(
            SiameseEncodersDataset(
                self.groups_train,
                nr_samples=self.nr_samples,  
                normalize=self.normalize,
                augmentations=self.augmentations,
                transform_override=self.transform_override,
                batch_sampling_mode=self.batch_sampling_mode,
                sr=self.sr,
                multi_epoch=self.multi_epoch,
            ),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            SiameseEncodersDataset(
                self.groups_eval,
                nr_samples=self.nr_samples,  
                normalize=self.normalize,
                augmentations={},  # No augmentations for validation
                batch_sampling_mode=self.batch_sampling_mode,
                sr=self.sr,
                multi_epoch=1,
            ),
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def prepare_data_end(self):
        print(
            f"Augmentations: {json.dumps(self.augmentations, sort_keys=True, indent=4)}"
        )
