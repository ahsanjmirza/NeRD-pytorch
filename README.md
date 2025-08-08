# PyTorch NeRD

This repository contains a PyTorch reimplementation of **NeRD (Neural Reflectance Decomposition)** originally proposed by Boss *et al.* in their TensorFlow codebase. This version focuses on a minimal, debloated codebase for clarity and ease of customization.

## Table of Contents

* [Features](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#features)
* [Paper](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#paper)
* [Setup](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#setup)
* [Dataset Preparation](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#dataset-preparation)
* [Training](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#training)
* [Inference](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#inference)
* [Project Structure](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#project-structure)
* [Configuration](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#configuration)
* [License](https://chatgpt.com/c/68965c42-2b24-8332-ac64-9a86824ab03f#license)

## Features

* Full PyTorch implementation of NeRD’s coarse and fine networks
* Spherical Gaussian illumination model for relighting
* Explicit BRDF autoencoder and normal estimation
* Debloated, modular code for easy extension
* Supports both varying and fixed illumination setups

## Paper

> **NeRD: Neural Reflectance Decomposition from Image Collections**
>
> Mark Boss, Raphael Braun, Varun Jampani, Jonathan T. Barron, Ce Liu, Hendrik P.A. Lensch.  CVPR 2021.
>
> [Project Page](https://markboss.me/publication/2021-nerd/)

## Setup

1. **Clone** this repository:
   ```bash
   git clone https://github.com/ahsanjmirza/pytorch-nerd.git
   cd pytorch-nerd
   ```
2. **Install** dependencies (recommend using a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **GPU** support is strongly recommended for performance.

## Dataset Preparation

We provide a helper script to preprocess your image collections, compute camera poses and bounds, and split data for training and validation.

1. **Directory structure** :

```
   experiments/
     └── MyScene/
         ├── images/          # Input RGB images (PNG/JPEG)
         ├── masks/           # Binary foreground masks (same resolution)
         └── config.json      # Scene-specific settings (see example)
```

1. **Run** the preparation script:

   ```bash
   python prepare_dataset.py \
       --exp_dir ./experiments/MyScene \
       --factor 8            # (optional) downsampling factor
       --recenter True       # (optional) recenter camera poses
       --spherify False      # (optional) adjust radial poses
       --visualize           # (flag)    save poses_plot.png
   ```

   * **`--exp_dir`** : Path to your scene folder
   * **`--factor`** : Integer downscale factor for faster experiments
   * **`--recenter`** ,  **`--spherify`** : Boolean flags for pose processing
   * **`--visualize`** : Outputs a `poses_plot.png` in the experiment directory
2. **Outputs** created under `experiments/MyScene`:

   * `train.npy`, `val.npy`: Indices for training and validation splits
   * `images_factor{F}.npy`, `masks_factor{F}.npy`: Preprocessed arrays
   * `poses_bounds.npy`: Camera-to-world and bounds for each view
   * `poses_plot.png` (if `--visualize`): 3D plot of camera poses

## Training

Launch the training loop for coarse and fine networks:

```bash
python train.py --exp_dir ./experiments/MyScene --config config.json
```

* Checkpoints (`weights.pth`) and logs are saved under the experiment directory.
* TensorBoard logging is supported via `--log_dir` flag.

### Configurable Options

Controlled via `config.json` (see example below):

* `train_batch_size`, `initial_learning_rate`, `total_steps`
* Sampling counts: `coarse_samples`, `fine_samples`
* Network widths and depths for coarse/fine models
* BRDF latent dimensionality and number of SG lobes
* Learning rate decay schedule

## Project Structure

```
prepare_dataset.py           # Preprocess images, masks, poses
train.py                     # Training loop for NeRD
inference.py                 # Mesh extraction and rendering
model/
  ├─ nerd.py                 # High-level NeRD wrapper
  ├─ coarse.py               # Coarse neural field
  ├─ fine.py                 # Fine BRDF decomposition
  ├─ brdf_autoencoder.py     # BRDF latent space autoencoder
  ├─ sg_illumination.py      # Spherical Gaussian lighting model
  ├─ positional_encoder.py   # Fourier feature embeddings
  └─ renderer.py             # Differentiable BRDF renderer
utils/
  ├─ dataflow.py             # Data loading & pose utilities
  ├─ dataloader.py           # PyTorch Dataset definitions
  ├─ losses.py               # Loss functions
  ├─ math_utils.py           # Vector and matrix helpers
  ├─ exposure_helper.py      # Photometric exposure utilities
  ├─ exif_helper.py          # EXIF metadata reader
  ├─ mean_sgs.npy            # Initial SG environment parameters
  └─ misc.py                 # Miscellaneous utilities

experiments/                 # Sample scenes and configs
requirements.txt             # Python dependencies
README.md                    # This overview
LICENSE                      # MIT License
```

## Configuration

See [`config.example.json`](https://chatgpt.com/c/config.example.json) for full details:

```json
{
  "factor": 8,
  "recenter": true,
  "spherify": false,
  "path_zflat": true,
  "train_batch_size": 4096,
  "initial_learning_rate": 5e-4,
  "decay_rate": 0.1,
  "decay_steps_upto": 200000,
  "total_steps": 300000,
  "coarse_samples": 64,
  "fine_samples": 128,
  "n_layers_coarse": 8,
  "d_filter_coarse": 256,
  "n_layers_fine": 8,
  "d_filter_fine": 256,
  "d_brdf_latent": 2,
  "n_sg_lobes": 24,
  "n_sg_condense": 16
}
```

## License

This code is released under the MIT License. See [LICENSE](https://chatgpt.com/c/LICENSE) for details.
