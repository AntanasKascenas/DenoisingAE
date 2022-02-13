#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib

import torch.nn.functional as F


def normalise_percentile(volume):
    """
    Normalise with scaling by 99 percentile max.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :, :] /= p_99

    return volume


def process_patient(path, target_path):
    flair = nib.load(path / f"{path.name}_flair.nii.gz").get_fdata()
    t1 = nib.load(path / f"{path.name}_t1.nii.gz").get_fdata()
    t1ce = nib.load(path / f"{path.name}_t1ce.nii.gz").get_fdata()
    t2 = nib.load(path / f"{path.name}_t2.nii.gz").get_fdata()
    labels = nib.load(path / f"{path.name}_seg.nii.gz").get_fdata()

    volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
    labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)

    patient_dir = target_path / f"patient_{patient_idx}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    volume = normalise_percentile(volume)

    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

    for slice_idx in range(fs_dim2, ls_dim2):
        low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
        low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))

        np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str, help="path to Brats2021 Training Data directory")

    args = parser.parse_args()

    datapath = Path(args.source)

    train_imgs = sorted(list((datapath).iterdir()))

    indices = list(range(len(train_imgs)))
    random.seed(0)
    random.shuffle(indices)

    n_train = int(len(indices) * 0.75)
    n_val = int(len(indices) * 0.05)
    n_test = len(indices) - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"Patients in train: {len(train_indices)}")
    print(f"Patients in val: {len(val_indices)}")
    print(f"Patients in test: {len(test_indices)}")

    for patient_idx in train_indices:
        path = train_imgs[patient_idx]

        target_path = Path(__file__).parent.parent / "data" / "brats2021_preprocessed" / "npy_train"

        process_patient(path, target_path)

    for patient_idx in val_indices:
        path = train_imgs[patient_idx]
        target_path = Path(__file__).parent.parent / "data" / "brats2021_preprocessed" / "npy_val"

        process_patient(path, target_path)

    for patient_idx in test_indices:
        path = train_imgs[patient_idx]
        target_path = Path(__file__).parent.parent / "data" / "brats2021_preprocessed" / "npy_test"

        process_patient(path, target_path)
