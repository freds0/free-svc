import torch
from torch.utils.data import Dataset
import numpy as np

from singer_identity.utils.core import (
    get_fragment_from_file,
    prepare_fn_groups_vocal,
    filter1_voice_wav,
)
from .augmentations import aug

import pytorch_lightning as pl
import random
import os


class BaseDictDataset(Dataset):
    """Base class for datasets that use a dictionary to store filepaths of audio files"""

    def __init__(
        self,
        groups: dict,
        nr_samples: int,
        normalize: bool = True,
        augmentations: dict = {},
        transform_override: bool = False,
        batch_sampling_mode: str = "sample_clips",
        sr: int = 44100,
        multi_epoch: int = 1,
    ):
        """
        Args:
            groups: Dictionary containing (group_name -> list of filenames) pairs. Group_name can be artist name for example
            nr_samples: Number of samples of fragment to be extracted on data loading.
                The audio fragment is extracted randomly from a sampled audio file (clip)
            normalize: Wether to normalize or not the extracted fragments
            augmentations: Dict containing augmentations to perform on positive examples and it's probabilities
            transform_override: If true, augmentations will contain a dict of transforms that will replace the positive augmentations loader
            batch_sampling_mode: If "sample_clips", clips are sampled randomly out of a list containig all clips on the data (default, unsupervised)
                                 If "sample_groups", first a group is sampled out of the list of groups, then a clip is sampled from that group (supervised)
            sr: Sample rate of audio files
            multi_epoch: gives the ilusion that the dataset has multi_epoch more items than it has allows for sampling the same group multiple times
        """
        super().__init__()

        self.groups = groups
        self.multi_epoch = multi_epoch
        self.nr_samples = nr_samples
        self.normalize = normalize
        self.transform_override = transform_override
        self.sr = sr

        self.augmentations = False
        if augmentations.get("enable", 0):
            if self.transform_override:
                raise ValueError(
                    "Transform override but augmentations passed are not the transforms"
                )
            if augmentations["enable"]:
                self.augmentations = augmentations
        if self.transform_override:
            self.augmentations = augmentations

        self.batch_sampling_mode = batch_sampling_mode
        self.prepare_dataset()

    
    def prepare_dataset(self):
        """Creates a inverse dictionary mapping filenames to group names"""
        self.groups_keys = list(self.groups.keys())
        self.inv_map = {fn: k for k, v in self.groups.items() for fn in v}
        self.inv_map_keys = list(self.inv_map.keys())
        self.inv_map_values = list(self.inv_map.values())
        self.data_len = (
            len(self.inv_map)
            if self.batch_sampling_mode == "sample_clips"
            else len(self.groups)
        )

    def __getitem__(self, item):
        item = item % self.data_len
        while True:
            try:
                fn, group_name = self.get_clip_and_group_name(item)  # Samples a clip
                # Additional get item
                result = self.getitem(item, file=fn, group_name=group_name)
                if result is None:
                    continue
                result = self.augment(result)  # Augmentations
                return self.return_data(
                    result, group_name, item
                )  # Adds extra values and return
            except AssertionError as e:
                raise e
            except Exception as e:
                raise e

    def __len__(self):
        return self.data_len * self.multi_epoch

    def getitem(self, item, file=None, group_name=None):
        """Returns data from a given item. Needs to be overriden by child classes"""
        raise NotImplementedError

    def get_fragment(self, fn):
        """Returns randomly sampled, normalized audio fragment of size self.nr_samples from file fn"""
        frag = get_fragment_from_file(
            fn, self.nr_samples, self.normalize, draw_random=True, sr=self.sr
        )
        if frag is None:
            print(
                f"Warning (get_fragment): could not get fragment from {fn}. Returning silence vector"
            )
            frag = torch.zeros(self.nr_samples)
        return frag

    def get_clip_and_group_name(self, item):
        """Samples from dataset
        Returns:
            fn: samples filename
            group_name: name of group it belongs to
        """

        if self.batch_sampling_mode == "sample_clips":
            fn = self.inv_map_keys[item]  # Sample a clip in the dataset
            group_name = self.inv_map_values[item]  # Gets clip artist
        elif self.batch_sampling_mode == "sample_groups":
            group_name = self.groups_keys[item]  # Sample an artist (group)
            selec_group = self.groups[group_name]  # Gets list of fns of the artist
            fn = selec_group[np.random.randint(len(selec_group))]
        else:
            raise ValueError(
                f"Invalid batch sampling mode. Value was {self.batch_sampling_mode}"
            )

        return fn, group_name

    def augment(self, data):
        """
        Performs augmentations described in dict self.augmentations on input data
        Can be overriden by transform_override, in which case  self.augmentations is
            expected to contain a dictionary whose values contains the transforms
            themselves to be applied
        """
        override = self.augmentations if self.transform_override else False
        return aug(np.cast["float32"](data), self.augmentations, override=override)

    def return_data(self, result=None, group_name=None, idx=None):
        return result


class BaseDataModule(pl.LightningDataModule):
    """Base class for data modules that use a dictionary to store filepaths of audio files"""

    def __init__(
        self,
        dataset_dirs: list = [],
        batch_size: int = 32,
        batch_size_val: int = 32,
        num_workers: int = 8,
        eval_frac: float = 0.1,
        group_name_is_folder: bool = False,
        group_by_artist: bool = False,
        verbose: bool = True,
        use_random_loader: bool = False,
        max_groups: int = -1,
        nr_samples: int = 176000,
        normalize: bool = True,
        augmentations: dict = {},
        transform_override: bool = False,
        batch_sampling_mode: str = "sample_clips",
        sr: int = 44100,
        multi_epoch: int = 1,
    ):
        """
        Args:
            dataset_dirs: List of directories where data is contained. Dirs will be scanned and
                the audio files will be added to the data dictionary
            batch_size: Train loader batch size
            batch_size_val: Validation loader batch size
            num_workers: Number of workers
            group_by_artist: If true, assumes each subfolder on dataset directory is a group.
                All files on this folder will be mapped to the same entry on the groups dictionary (either folder name or 'unknown',
                depending on group_name_is_folder).
                If false, each file will be mapped to it's own group
            group_name_is_folder: If true, the dict key of a group will be the folder name
            eval_frac: Fraction of data groups to be separated as validation set
            use_random_loader: Loads a random loader (for debug purposes)
            max_groups: if > 1, selects a random subset of the data dictionary containing max_groups entries
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.dataset_dirs = dataset_dirs
        self.dataset_names = []
        self.group_name_is_folder = group_name_is_folder
        self.group_by_artist = group_by_artist
        self.eval_frac = eval_frac
        self.groups = []
        self.group_names = []  # maybe to register buffer
        self.num_workers = num_workers
        self.verbose = verbose
        self.use_random_loader = use_random_loader
        self.max_groups = max_groups
        self.dataset_kwargs = {
            "nr_samples": nr_samples,
            "normalize": normalize,
            "augmentations": augmentations,
            "transform_override": transform_override,
            "batch_sampling_mode": batch_sampling_mode,
            "sr": sr,
            "multi_epoch": multi_epoch,
        }

    def prepare_data(self):
        self.prepare_data_end()

    def setup(self, stage=None):
        assert len(self.dataset_dirs) > 0, "No dataset directories provided"
        self._process_dataset_dirs()
        self._merge_groups()
        self._perform_train_val_split()

    def _process_dataset_dirs(self):
        for dataset in self.dataset_dirs:
            dataset_name = os.path.basename(dataset)

            groups = self._prepare_groups(dataset)
            if groups:
                self.groups.append(groups)
                self.dataset_names.append(dataset_name)

        self.group_names_separate_datasets = [
            list(dataset.keys()) for dataset in self.groups
        ]
        self._print_dataset_files_info()

    def _prepare_groups(self, dataset):
        return prepare_fn_groups_vocal(
            dataset,
            groups=None,
            filter_fun_level1=filter1_voice_wav,
            group_name_is_folder=self.group_name_is_folder,
            group_by_artist=self.group_by_artist,
        )

    def _print_dataset_files_info(self):
        for i, dataset in enumerate(self.dataset_names):
            print(
                f"Number of files in dataset {dataset}: {self._count_files_in_dict(self.groups[i])}, split into {len(self.groups[i].keys())} artists"
            )

    def _count_files_in_dict(self, dic):
        return sum([len(group) for group in list(dic.values())])

    def _count_elements_in_dict_split(self, dic, keys_subset):
        return sum([len(dic[key]) for key in keys_subset])

    def _merge_groups(self):
        """Merges groups from different datasets into one large dataset"""
        groups = {}
        for i, group in enumerate(self.groups):
            if self.max_groups > 1:
                self.group_names_separate_datasets[i] = random.sample(
                    self.group_names_separate_datasets[i], self.max_groups
                )

            for key in self.group_names_separate_datasets[i]:
                if groups.get(key, 0):
                    groups[key] = groups[key] + group[key]
                else:
                    groups[key] = group[key]
            group = {key: group[key] for key in self.group_names_separate_datasets[i]}

        self.groups = groups
        self.group_names = list(self.groups.keys())
        np.random.shuffle(self.group_names)

        for subset_group_names in self.group_names_separate_datasets:
            np.random.shuffle(subset_group_names)

        self.n_files = self._count_files_in_dict(self.groups)
        self.n_groups = len(self.group_names)
        print(
            f"Number of files in full merged dataset: {self.n_files}, split into {self.n_groups} artists"
        )

    def _perform_train_val_split(self):
        self.eval_split = int(len(self.group_names) * self.eval_frac)
        self.eval_splits = [
            int(len(subset_group_names) * self.eval_frac)
            for subset_group_names in self.group_names_separate_datasets
        ]
        if self.eval_split < 1:
            print("Warning: current evaluation split selects less than one group.")
            self.eval_split = 1

        groups_train = {}
        groups_eval = {}
        for i, subset_group_names in enumerate(self.group_names_separate_datasets):
            groups_train.update(
                dict(
                    (k, self.groups[k])
                    for k in subset_group_names[self.eval_splits[i] :]
                )
            )
            groups_eval.update(
                dict(
                    (k, self.groups[k])
                    for k in subset_group_names[: self.eval_splits[i]]
                )
            )

        self.groups_train = groups_train
        self.groups_eval = groups_eval

        self.n_files_train = self._count_elements_in_dict_split(
            self.groups, groups_train.keys()
        )
        self.n_files_eval = self._count_elements_in_dict_split(
            self.groups, groups_eval.keys()
        )
        self._print_train_val_split_info()

    def _print_train_val_split_info(self):
        if self.verbose and self.eval_frac < 1:
            print(
                f"Size train (groups) : {self._count_files_in_dict(self.groups_train)}, eval: {self._count_files_in_dict(self.groups_eval)}"
            )
            print(f"Size train (files): {self.n_files_train}")
            print(f"Size eval (files): {self.n_files_eval}")

    def prepare_data_end(self):
        return
