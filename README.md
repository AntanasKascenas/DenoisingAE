# Denoising Autoencoders for Unsupervised Anomaly Detection

## Introduction

This repository hosts the code that implements, trains and evaluates denoising autoencoders described in [https://openreview.net/forum?id=Bm8-t_ggzPD](https://openreview.net/forum?id=Bm8-t_ggzPD)

## Usage

'src/data_preprocessing.py' to process the BraTS2021 data.

'src/denoising.py' to train a denoising autoencoder model.

'src/evaluate.py' to evaluate a trained model.

## Requirements
Dependency requirements can be found in 'environment.yml'

Use a conda environment to install the required libraries:
``
conda env create -f environment.yml
``