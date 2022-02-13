#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np


class PatientDatasetNP(torch.utils.data.Dataset):

    def __init__(self, patient_dir: Path, process_fun=None, id=None, skip_condition=None):

        self.patient_dir = patient_dir
        # Make sure the slices are correctly sorted according to the slice number in case we want to assemble
        # "pseudo"-volumes later.
        self.slice_paths = sorted(list(patient_dir.iterdir()), key=lambda x: int(x.name[6:-4]))
        self.process = process_fun
        self.skip_condition = skip_condition
        self.id = id
        self.len = len(self.slice_paths)
        self.idx_map = {x: x for x in range(self.len)}

        if self.skip_condition is not None:

            # Try and find which slices should be skipped and thus determine the length of the dataset.
            valid_indices = []
            for idx in range(self.len):
                with np.load(self.slice_paths[idx]) as data:
                    if self.process is not None:
                        data = self.process(**data)
                    if not skip_condition(data):
                        valid_indices.append(idx)
            self.len = len(valid_indices)
            self.idx_map = {x: valid_indices[x] for x in range(self.len)}

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data = np.load(self.slice_paths[idx])

        if self.process is not None:
            data = self.process(**data)
        return data

    def __len__(self):
        return self.len


class BrainDataset(torch.utils.data.Dataset):

    def __init__(self, dataset="brats2021", split="val", n_tumour_patients=None, n_healthy_patients=None,
                 skip_healthy_s_in_tumour=False,  # whether to skip healthy slices in "tumour" patients
                 skip_tumour_s_in_healthy=True,  # whether to skip tumour slices in healthy patients
                 seed=0):

        self.rng = random.Random(seed)

        if dataset == "brats2021":
            datapath = Path(__file__).parent.parent / "data" / "brats2021_preprocessed"

            train_path = datapath / "npy_train"
            val_path = datapath / "npy_val"
            test_path = datapath / "npy_test"

        else:
            raise ValueError(f"dataset {dataset} unknown")

        if split == "train":
            path = train_path
        elif split == "val":
            path = val_path
        elif split == "test":
            path = test_path
        else:
            raise ValueError(f"split {split} unknown")

        # Slice skip conditions:
        threshold = 0
        self.skip_tumour = lambda item: item[1].sum() > threshold
        self.skip_healthy = lambda item: item[1].sum() <= threshold

        def process(x, y, coords=None): # TODO remove coords here.
            # treat all tumour classes as one for anomaly detection purposes.
            y = y > 0.5

            return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float()

        patient_dirs = sorted(list(path.iterdir()))
        self.rng.shuffle(patient_dirs)

        assert ((n_tumour_patients is not None) or (n_healthy_patients is not None))
        self.n_tumour_patients = n_tumour_patients if n_tumour_patients is not None else len(patient_dirs)
        self.n_healthy_patients = n_healthy_patients if n_healthy_patients is not None else len(patient_dirs) - self.n_tumour_patients

        # Patients with tumours
        self.patient_datasets = [PatientDatasetNP(patient_dirs[i], process_fun=process, id=i,
                                                  skip_condition=self.skip_healthy if skip_healthy_s_in_tumour else None)
                                 for i in range(self.n_tumour_patients)]

        # + only healthy slices from "healthy" patients
        self.patient_datasets += [PatientDatasetNP(patient_dirs[i],
                                                   skip_condition=self.skip_tumour if skip_tumour_s_in_healthy else None,
                                                   process_fun=process, id=i) for i in range(self.n_tumour_patients, self.n_tumour_patients + self.n_healthy_patients)]

        self.dataset = ConcatDataset(self.patient_datasets)

    def __getitem__(self, idx):
        x, gt = self.dataset[idx]
        return x, gt

    def __len__(self):
        return len(self.dataset)


