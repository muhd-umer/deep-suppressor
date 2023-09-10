# DeepSuppressor
DeepSuppressor is a deep learning-based speech denoiser which can significantly improve the quality of speech signals. The denoiser is trained on a large dataset of clean and noisy speech signals, and it can be used to denoise speech signals in real time or offline.

## Installation
To get started with this project, follow the steps below:

**Clone the repository**
- Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/muhd-umer/deep-suppressor.git
```

**Create a new virtual environment**
- It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects. To create a new virtual environment, run the following command:
```shell
conda env create -f environment.yml
```

- Alternatively, you can use `mamba` (faster than conda) package manager to create a new virtual environment:
```shell
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

**Install the dependencies**
- Activate the newly created environment:
```shell
conda activate deep-suppressor
```

- Install PyTorch (Stable 2.0.1):
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
