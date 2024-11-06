# General Overview

This repository contains scripts for full-field material response prediction and uncertainty quantification (UQ) using Bayesian neural networks. The prediction between the input microstructure and the output stress field is cast as an image-to-image regression problem, utilizing a modified U-net neural network alongside three UQ methods, all of which are derived from the `Unet` base class.

## Implemented Methods

The following methods are supported:

- **Deterministic**: Standard neural network training.
- **MCD**: Monte Carlo Dropout.
- **BBB**: BayesByBackprop.
- **HMC**: Hamiltonian Monte Carlo.

## Main Functions

The primary functions in the scripts include:

- `ModelTrain`: Used for model training.
- `ModelPredict`: Used for making predictions.
- `ModelPlot`: Used for visualizing the results.

## Data

This framework has been evaluated with two types of material datasets:

1. **Fiber Reinforced Composite**
2. **Polycrystalline Material System**

The datasets can be downloaded from the following [link](https://livejohnshopkins.sharepoint.com/:f:/r/sites/JHUDataArchive/Shared%20Documents/ShieldsM_JHRDataRepository_20241031/data?csf=1&web=1&e=CMc09P). It is recommended to download each dataset separately to avoid large file sizes and potential corruption.

## Configuration Parameters

Key configuration parameters are stored in the `configs` folder, which defines the model's behavior. The U-net architecture is generated automatically based on the selected parameters. Below are some essential parameters:

- **method**: Specifies the approach, with options including `Frequentist`, `MCD`, `BBB`, `BBB_LRT`, and `HMC`.
- **case**: Creates organized subdirectories in the `data` and `trained_models` folders.
- **patience**: Sets the training patience (number of epochs without improvement).
- **num_samples**: Number of data points in the dataset.
- **batch_size**: Batch size during training (for MCD, Frequentist, BBB, BBB_LRT).
- **n_target_ch**: Number of target output stress channels (default is 1).
- **nfilters**: Encoding-decoding filter sizes.
- **kernel_size**: Convolutional layer kernel size.
- **num_samples_bayes**: Number of realizations for MCD and BBB during UQ prediction.
- **num_samples_hmc**: Number of samples for HMC.
- **L**: HMC trajectory length.
- **step_size**: HMC step size.
- **device**: Computation device (`cuda`, `mps`, `cpu`, etc.).
- **drop_idx_en**: denote the positions of the dropout layers in the encoding and decoding paths, respectively

> **Note**: Preloaded models should be run on the same device they were trained on to avoid compatibility issues.

## Steps for Using the Fiber Dataset

1. Download the data and create a `data/data_fiber` folder.
2. Specify `"fiber"` in the `config.json`.
3. Adjust parameters in the configuration files according to the selected method.
4. Run the training script with:
   - `python train_model` for model training
   - `python predict_model` for model predictions using the trained model
   - `python plot_model` to visualize the results
  These steps can be repeated for Deterministic, MCD, BBB and HMC. MCD and HMC use pretrained models so there is no need to be run from scratch.

## References

1. Pasparakis, G.D., Graham-Brady, L., and Shields, M.D. (2024). *Bayesian neural networks for predicting uncertainty in full-field material response*. arXiv preprint arXiv:2406.14838.
2. Shridhar, K., Laumann, F., and Liwicki, M. (2019). *A comprehensive guide to Bayesian convolutional neural networks with variational inference*. arXiv preprint arXiv:1901.02731.
3. Cobb, A.D., and Jalaian, B. (2021, December). *Scaling Hamiltonian Monte Carlo inference for Bayesian neural networks with symmetric splitting*. In *Uncertainty in Artificial Intelligence* (pp. 675-685). PMLR.
