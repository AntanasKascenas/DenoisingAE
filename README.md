# Denoising Autoencoders for Unsupervised Anomaly Detection

## Introduction

This repository hosts the code that implements, trains and evaluates denoising autoencoders described in:

Kascenas, A., Pugeault, N. and O'Neil, A.Q., 2021. **Denoising Autoencoders for Unsupervised Anomaly Detection in Brain MRI.**
[https://openreview.net/forum?id=Bm8-t_ggzPD](https://openreview.net/forum?id=Bm8-t_ggzPD)

&nbsp;

![DAE system diagram](images/system_diagram.png)
The denoising autoencoder anomaly detection pipeline. During training (top), noise is added to the foreground of the healthy image, and the network is trained to reconstruct the original image.
At test time (bottom), the pixelwise post-processed reconstruction error is used as the anomaly score.
## Usage

1. Use 'src/data_preprocessing.py' to process the BraTS2021 data. See [http://www.braintumorsegmentation.org/](http://www.braintumorsegmentation.org/) for requesting/downloading the data.
Command line arguments:
    * -s, --source | A path pointing to the unzipped directory of BraTS2021 Training data. E.g. /data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021

2. Use 'src/denoising.py' to train a denoising autoencoder model.
Command line arguments:
    * -id, --identifier | Arbitrary model name to save under. Defaults to "model".
    * -nr, --noise_res | Noise resolution to use. Defaults to "16".
    * -ns, --noise_std | Noise magnitude to use. Defaults to "0.2".
    * -s,  --seed | Determines the data loading order. Defaults to "0".
    * -bs, --batch_size | Determines the batch size used during training of the DAE. Defaults to "16".

3. Use 'src/evaluate.py' to evaluate a trained model. Command line arguments:
    * -id, --identifier | Name of the model to load and evaluate. Defaults to "model".
    * -s, --split | Split of the dataset to evaluate on. One of "train", "val", "test". Defaults to "test".
    * -cc, --use_cc | Whether to use connected component filtering. Defaults to "True".

## Requirements
Dependency requirements can be found in 'environment.yml'

Use a conda environment to install the required libraries:
`$ conda env create -f environment.yml`