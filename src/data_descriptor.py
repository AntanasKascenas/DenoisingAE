#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from typing import Optional

import torch

from data import BrainDataset


class DataDescriptor:

    def __init__(self, n_workers=6, batch_size=16, **kwargs):

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataset_cache = {}

    def get_dataset(self, split: str):
        raise NotImplemented("get_dataset needs to be overridden in a subclass.")

    def get_dataset_(self, split: str, cache=True, force=False):
        if split not in self.dataset_cache or force:
            dataset = self.get_dataset(split)
            if cache:
                self.dataset_cache[split] = dataset
            return dataset
        else:
            return self.dataset_cache[split]

    def get_dataloader(self, split: str):
        dataset = self.get_dataset_(split, cache=True)

        shuffle = True if split == "train" else False
        drop_last = False if len(dataset) < self.batch_size else True

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last,
                                                 num_workers=self.n_workers)

        return dataloader


class BrainAEDataDescriptor(DataDescriptor):

    def __init__(self, dataset="brats2021", n_train_patients: Optional[int] = None, n_val_patients: Optional[int] = 20, seed: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.seed = seed
        self.dataset = dataset
        self.n_train_patients = n_train_patients
        self.n_val_patients = n_val_patients

    def get_dataset(self, split: str):
        assert split in ["train", "val"]  # "test" should not be used through the DataDescriptor interface in this case.

        if split == "train":
            n_healthy_patients = self.n_train_patients
        elif split == "val":
            n_healthy_patients = self.n_val_patients

        seed = 0 if split == "val" else self.seed
        dataset = BrainDataset(split=split, dataset=self.dataset, n_tumour_patients=0,
                               n_healthy_patients=n_healthy_patients,
                               seed=seed)

        return dataset



