# HMC-VAE
This repository contains code for hierarchical convolutional variational autoencoders with Hamiltonian Monte Carlo, which accompanies my MPhil thesis at University of Cambridge.

### Directory Structure
The reposiroty contains 7 Python files:
```
models/hmc.py      # Basic version of Hamiltonian Monte Carlo
models/blocks.py   # Conv/ConvTranspose building blocks for the model
models/vae.py      # VAE with hierarchical latent variables
models/hmc_vae.py  # VAE with hierarchical latent variables and HMC
image_dataset.py   # Download and preprocess image datasets
utils.py           # Utility for turning on/off parts of a model
train.py           # Main entry point for traning
```
And my thesis `thesis.pdf` that details theoretical background, implementation details, and experimental results.

### Example
The command below trains a model with 2 latent layers each containing 30 units, and 64 filters, on the dataset CIFAR10 (which is stored in `./data`), for 100 epochs (100 variational epochs and hard-coded 3 HMC epochs), with the test set evaluated after every 1 epoch, where both trained model and test set scores are written to `/path/to/save/`.
```
python3 train.py --data_name CIFAR10 --data_root ./data \
    --hidden_channels 64 --epochs 100 --eval_interval 1 \
    --savedir /path/to/save/ --latent_dims 30 30
```

### Architecture Visualization (1 latent layer)
Encoder

<img width="800" alt="encoder" src="https://user-images.githubusercontent.com/22922351/185190687-f2b5f2d5-420f-49a1-8203-3000ec2ea54a.png">

Decoder

<img width="800" alt="decoder" src="https://user-images.githubusercontent.com/22922351/185190737-81eb13d0-946d-4262-a66d-e48f710885f3.png">


### Citation
```
@Misc{HMC-VAE,
    author = {Haoran Peng},
    title = {Outlier Detection with Hierarchical VAEs and Hamiltonian Monte Carlo},
    year = {2022},
    url = "https://github.com/GavinPHR/HMC-VAE"
}
```
