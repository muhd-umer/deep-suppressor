"""
Helper functions for the AudioDataset class.
"""

import torch
import matplotlib.pyplot as plt


def load_data(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Returns a DataLoader for the dataset.

    Args:
        dataset (AudioDataset): Dataset to load.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of worker threads to use for loading the data. Defaults to 0.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def visualize_spectrogram(dataloader, num_samples=4):
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


def test_dataloader(dataloader):
    """
    Iterates over the dataloader and prints the shapes of the first 4 samples.

    Args:
        dataloader (DataLoader): DataLoader to iterate over.
    """
    for i, (data, target) in enumerate(dataloader):
        if i >= 2:
            break
        print(f"Sample {i+1} - Data shape: {data.shape}, Target shape: {target.shape}")
