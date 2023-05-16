"""
Custom dataloader for the clean, noisy, and mixed audio data
"""

import os
import os.path as osp
import glob
import torch
import torchaudio
from typing import Tuple
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    PyTorch Dataset class for loading audio files and their corresponding spectrograms.

    Args:
        files (list): List of file paths to load.
        train (bool, optional): Whether the dataset is for training or
        validation. Defaults to True.
    """

    def __init__(self, files, train=True):
        """
        Initializes the AudioDataset.

        Args:
            files (list): List of file paths to load.
            train (bool, optional): Whether the dataset is for training or
            validation. Defaults to True.
        """
        self.files = files
        self.train = train
        self.sr = 8000
        self.speech_length_pix_sec = 27e-3
        self.total_length = 3.6
        self.trim_length = 28305
        self.n_fft = 255
        self.frame_length = 255
        self.frame_step = int(445 / 4)

        # Define the absolute paths of the data directories
        self.DATA_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data")
        )
        self.NOISE_DIR = os.path.join(self.DATA_DIR, "noise_files")
        self.SPEECH_DIR = os.path.join(self.DATA_DIR, "speech_files")
        self.noise_files = glob.glob(os.path.join(self.NOISE_DIR, "*.wav"))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the item at the given index.

        Args:
            idx (int): Index of the item to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the corrupted
            spectrogram and the clean spectrogram.
        """
        filepath = self.files[idx]
        wav = self.load_wav(filepath)  # (28400,)
        wav_corr, wavclean = self.white_noise(wav)  # (28400,), (28400,)
        wav_corr, wavclean = self.urban_noise(wav_corr, wavclean)  # (28400,), (28400,)
        spectrogram_corr, spectrogram = self.convert_to_spectrogram(
            wav_corr, wavclean
        )  # (128, 256), (128, 256)
        if not self.train:
            spectrogram_corr, spectrogram = self.expand_dims(
                spectrogram_corr, spectrogram
            )  # (1, 128, 256), (1, 128, 256)
            # Return the spectrograms without augmentation
            return spectrogram_corr, spectrogram

        spectrogram_corr, spectrogram = self.spectrogram_abs(
            spectrogram_corr, spectrogram
        )  # (128, 256), (128, 256)
        spectrogram_corr, spectrogram = self.expand_dims(
            spectrogram_corr, spectrogram
        )  # (1, 128, 256), (1, 128, 256)
        spectrogram_corr, spectrogram = self.augment(
            spectrogram_corr, spectrogram
        )  # (1, 128, 256), (1, 128, 256)
        # Return the augmented spectrograms
        return spectrogram_corr, spectrogram

    def load_wav(self, filename: str) -> torch.Tensor:
        """
        Loads a WAV file and returns its waveform.

        Args:
            filename (str): Path to the WAV file.

        Returns:
            torch.Tensor: Waveform of the WAV file.
        """
        waveform, _ = torchaudio.load(filename)  # type: ignore
        waveform = waveform.squeeze()
        return waveform

    def white_noise(self, data, factor=0.03):
        """
        Adds white noise to the given waveform.

        Args:
            data (torch.Tensor): Waveform to add noise to.
            factor (float, optional): Factor to scale the noise amplitude by.
            Defaults to 0.03.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the noisy waveform
            and the original waveform.
        """
        noise_amp = factor * torch.max(data) * torch.randn(1)
        corr_data = data + noise_amp * torch.randn(data.shape)
        return corr_data, data

    def urban_noise(self, corr_data, data, factor=0.4, sr=None):
        """
        Adds urban noise to the given waveform.

        Args:
            corr_data (torch.Tensor): Noisy waveform to add urban noise to.
            data (torch.Tensor): Original waveform.
            factor (float, optional): Factor to scale the noise amplitude by.
            Defaults to 0.4.
            sr (int, optional): Sample rate of the waveform. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the waveform with
            urban noise and the original waveform.
        """
        noisefile = self.noise_files[torch.randint(len(self.noise_files), (1,))]
        noisefile = self.load_wav(noisefile)
        mixed = (
            noisefile * factor * torch.max(corr_data) / torch.max(noisefile) + corr_data
        )
        return mixed, data

    def convert_to_spectrogram(self, wav_corr, wavclean):
        """
        Converts the given waveforms to spectrograms.

        Args:
            wav_corr (torch.Tensor): Noisy waveform.
            wavclean (torch.Tensor): Original waveform.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the spectrogram of
            the noisy waveform and the spectrogram of the original waveform.
        """
        spectrogram_corr = torch.stft(
            wav_corr,
            n_fft=self.n_fft,
            hop_length=self.frame_step,
            win_length=self.frame_length,
            return_complex=True,
        )
        spectrogram = torch.stft(
            wavclean,
            n_fft=self.n_fft,
            hop_length=self.frame_step,
            win_length=self.frame_length,
            return_complex=True,
        )
        return spectrogram_corr, spectrogram

    def spectrogram_abs(self, spectrogram_corr, spectrogram):
        """
        Computes the absolute value of the given spectrograms.

        Args:
            spectrogram_corr (torch.Tensor): Noisy spectrogram.
            spectrogram (torch.Tensor): Original spectrogram.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the absolute value
            of the noisy spectrogram and the absolute value of the original spectrogram.
        """
        spectrogram = torch.abs(spectrogram)
        spectrogram_corr = torch.abs(spectrogram_corr)
        return spectrogram_corr, spectrogram

    def augment(self, spectrogram_corr, spectrogram):
        """
        Applies data augmentation to the given spectrograms.

        Args:
            spectrogram_corr (torch.Tensor): Noisy spectrogram.
            spectrogram (torch.Tensor): Original spectrogram.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the augmented noisy
            spectrogram and the augmented original spectrogram.
        """
        # Apply frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        spectrogram_corr = freq_mask(spectrogram_corr)

        # Apply time masking
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
        spectrogram_corr = time_mask(spectrogram_corr)
        return spectrogram_corr, spectrogram

    def expand_dims(self, spectrogram_corr, spectrogram):
        """
        Adds a dimension to the given spectrograms.

        Args:
            spectrogram_corr (torch.Tensor): Noisy spectrogram.
            spectrogram (torch.Tensor): Original spectrogram.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the expanded noisy
            spectrogram and the expanded original spectrogram.
        """
        spectrogram_corr = spectrogram_corr.unsqueeze(0)
        spectrogram = spectrogram.unsqueeze(0)
        return spectrogram_corr, spectrogram
