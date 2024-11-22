# Overview

This repository contains the scripts for full-field material response prediction and uncertainty quantification (UQ) using Bayesian neural networks. The prediction between the input microstructure and the output stress field is cast as an image-to-image regression problem, utilizing a modified U-net neural network alongside three UQ methods, all of which are derived from the `Unet` base class.

## Implemented methods

The following methods are supported:

- **Deterministic**: Standard neural network training.
- **MCD**: Monte Carlo Dropout.
- **BBB**: BayesByBackprop.
- **HMC**: Hamiltonian Monte Carlo.

## Main functions

The primary functions in the scripts include:

- `ModelTrain`: Used for model training.
- `ModelPredict`: Used for making predictions.
- `ModelPlot`: Used for visualizing the results.

## Data

This framework has been evaluated with two types of material datasets:

1. **Fiber Reinforced Composite**
2. **Polycrystalline Material System**

The datasets can be downloaded from the following [link](https://doi.org/10.7281/T1GCHQPY). It is recommended to download each dataset separately to avoid large file sizes and potential corruption.

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

## Steps for reproducing the results
### Fiber reinforced composite
1. Download the data and create a `data/data_fiber` folder.
2. Specify `"fiber"` in the `config.json`.
3. Adjust parameters in the configuration files according to the selected method.
4. Run the training script with:
   - `python train_model` for model training
   - `python predict_model` for model predictions using the trained model
   - `python plot_model` to visualize the results
  These steps can be repeated for Deterministic, MCD, BBB. To run the HMC training, run `train_model_HMC.sh`. This requires a pre-trained deterministic model to be available. The resulting `*.pkl` files can be downloaded from this [link](https://doi.org/10.7281/T1GCHQPY) and should be stored in the `trained_models_fiber/case_1`. The MCD can also be run with a pre-trained model directly to perform predictions and plotting.
### Polycrystalline material system 
Same procedure as above. Unfortunately, the HMC cannot be run efficiently. All the trained weights are available in `trained_models_polycrystalline_2D/case_1` to reproduce the plots.

## References

* [Pasparakis, G.D., et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0045782524007400)
* [Shridhar, K., et al. (2019)](https://arxiv.org/abs/1901.02731)
* [Cobb, A.D., and Jalaian, B. (2021)](https://proceedings.mlr.press/v161/cobb21a.html)

## Aknowledgements
The authors gratefully acknowledge Dr. Ashwini Gupta and Indrashish Saha for providing the data and valuable insights regarding the fiber-reinforced composite material.

## Author
[George Pasparakis](https://scholar.google.com/citations?user=kPANZZQAAAAJ&hl=en)
