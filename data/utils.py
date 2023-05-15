"""
Helper functions for the AudioDataset class.
"""

import torch
import os.path as osp
from torch.utils.data import DataLoader
from .dataset import AudioDataset
import matplotlib.pyplot as plt
import glob

plt.style.use("default")


def get_dataloader(
    speech_dir, batch_size=16, split_ratio=0.2, shuffle=True, test=False
):
    """
    Returns train, validation, and optionally test DataLoader for the dataset.

    Args:
        speech_dir (str): Directory containing speech files.
        batch_size (int, optional): Batch size. Defaults to 32.
        split_ratio (float, optional): Ratio of validation set size to total dataset size. Defaults to 0.2.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        test (bool, optional): Whether to return a DataLoader for the test set. Defaults to False.

    Returns:
        tuple or DataLoader: A tuple of three DataLoaders for the train, validation, and test sets if `test` is True, otherwise a tuple of two DataLoaders for the train and validation sets.
    """
    speech_files = glob.glob(osp.join(speech_dir, "*.wav"))
    split_index = int(len(speech_files) * split_ratio)
    train_files = speech_files[split_index:]
    val_files = speech_files[:split_index]

    train_dataset = AudioDataset(train_files)
    val_dataset = AudioDataset(val_files)
    test_dataset = AudioDataset(val_files, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    if test:
        test_dataloader = DataLoader(test_dataset)
        return test_dataloader

    return train_dataloader, val_dataloader


def visualize_spectrogram(dataloader, num_samples=2):
    """
    Visualizes the spectrograms of input and target samples in the dataloader.

    Args:
        dataloader (DataLoader): DataLoader containing the spectrograms.
    """
    data, target = next(iter(dataloader))
    for j, (d, t) in enumerate(zip(data, target)):
        if j > num_samples - 1:
            break
        _, axs = plt.subplots(1, 2, figsize=(10, 8))
        axs[0].set_title(f"Input Spectrogram {j+1}")
        axs[0].imshow(torch.log(d[0]).numpy(), cmap="magma")
        axs[1].set_title(f"Corrupted Spectrogram {j+1}")
        axs[1].imshow(torch.log(t[0]).numpy(), cmap="magma")
        plt.show()


def dataloader_sampler(dataloader, num_samples=4):
    """
    Iterates over the dataloader and prints the shapes of the first 4 samples.

    Args:
        dataloader (DataLoader): DataLoader to iterate over.
    """
    for i, (data, target) in enumerate(dataloader):
        if i > num_samples - 1:
            break
        print(f"Sample {i+1} - Data shape: {data.shape}, Target shape: {target.shape}")
