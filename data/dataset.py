"""
Custom dataloader for the clean, noisy, and mixed audio data
"""

import os
import glob
import torch
import torchaudio
from typing import Tuple
from torch.utils.data import Dataset
import librosa
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from scipy.interpolate import interp1d

rs = RandomState(MT19937(SeedSequence(42)))


class AudioDataset(Dataset):
    """
    PyTorch Dataset class for loading audio files and their corresponding spectrograms.

    Args:
        files (list): List of file paths to load.
        train (bool, optional): Whether the dataset is for training or
        validation. Defaults to True.
        sample_only (bool, optional): Whether to return only the clean
    """

    def __init__(self, files, train=True, sample_only=False):
        """
        Initializes the AudioDataset.

        Args:
            files (list): List of file paths to load.
            train (bool, optional): Whether the dataset is for training or
            sample_only (bool, optional): Whether to return only the clean spectrogram.
            validation. Defaults to True.
        """
        self.files = files
        self.train = train
        self.sample_only = sample_only
        self.sr = 16000
        self.speech_length_pix_sec = 27e-3
        self.total_length = 3.6
        self.trim_length = 56800
        self.n_fft = 510
        self.frame_length = 510
        self.frame_step = 112 - 1

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

    def __getitem__(self, idx: int):
        """
        Returns the item at the given index.

        Args:
            idx (int): Index of the item to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the corrupted
            spectrogram and the clean spectrogram.
        """
        filepath = self.files[idx]
        wav = self.preprocess(filepath)  # (56800,)
        wav_corr, wavclean = self.white_noise(wav)  # (56800,), (56800,)
        wav_corr, wavclean = self.urban_noise(wav_corr, wavclean)  # (56800,), (56800,)
        # Convert to torch tensors
        wav_corr = torch.from_numpy(wav_corr).float()
        wavclean = torch.from_numpy(wavclean).float()
        spectrogram_corr, spectrogram = self.convert_to_spectrogram(
            wav_corr, wavclean
        )  # (128, 256), (128, 256)
        if self.sample_only:
            return spectrogram

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

    def preprocess(
        self,
        filepath,
        fixed_start=False,
    ):
        """
        Preprocesses the audio file at the given filepath by trimming it to a
        fixed length and converting it to a PyTorch tensor.

        Args:
            filepath (str): The path to the audio file to preprocess.
            fixed_start (int): If specified, the start index of the audio to
            trim to. Otherwise, a random start index is chosen.

        Returns:
            torch.Tensor: The preprocessed audio as a PyTorch tensor.
        """
        wav, _ = librosa.load(filepath, sr=self.sr)
        # wav, _ = librosa.effects.trim(wav, top_db=38)
        size = wav.shape[0]
        if size < self.trim_length:
            wav = np.pad(wav, (0, self.trim_length - size), "constant")
        elif size - self.trim_length - 1 > 0:
            random_start = rs.randint(0, size - self.trim_length - 1)
            wav = wav[random_start : random_start + self.trim_length]
        else:
            wav = wav[: self.trim_length]

        return wav

    def preprocess_torch(self, filepath: str) -> torch.Tensor:
        """
        Preprocesses a WAV file using PyTorch.

        Args:
            filepath (str): The path to the WAV file.

        Returns:
            torch.Tensor: The preprocessed audio data as a PyTorch tensor.
        """
        waveform = self.load_wav(filepath)
        waveform = waveform.squeeze()
        waveform = waveform[: self.trim_length]
        zero_padding = torch.zeros(self.trim_length - waveform.shape[0])
        waveform = torch.cat([zero_padding, waveform], 0)
        return waveform

    def white_noise(self, data, factor=0.03):
        """
        Adds white noise to the given waveform.

        Args:
            data (numpy.ndarray): Waveform to add noise to.
            factor (float, optional): Factor to scale the noise amplitude by.
                Defaults to 0.03.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the noisy waveform
                and the original waveform.
        """
        noise_amp = factor * np.max(data) * np.random.randn(1)
        corr_data = data + noise_amp * np.random.randn(data.shape[0])
        return corr_data, data

    def urban_noise(self, corr_data, data, factor=0.4):
        """
        Adds urban noise to the given waveform.

        Args:
            corr_data (numpy.ndarray): Noisy waveform to add urban noise to.
            data (numpy.ndarray): Original waveform.
            factor (float, optional): Factor to scale the noise amplitude by.
                Defaults to 0.4.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the waveform with
                urban noise and the original waveform.
        """
        noisefile = self.noise_files[np.random.randint(len(self.noise_files))]
        noisefile, _ = librosa.load(noisefile, sr=self.sr)

        if len(noisefile) < len(corr_data):
            x_old = np.linspace(0, 1, len(noisefile))
            x_new = np.linspace(0, 1, len(corr_data))
            f = interp1d(x_old, noisefile, kind="cubic")
            noisefile = f(x_new)
            noise_length = len(noisefile)
            if noise_length < len(corr_data):
                pad_length = len(corr_data) - noise_length
                noise = np.random.normal(0, 1, pad_length)
                noisefile = np.concatenate((noisefile, noise))
        elif len(noisefile) > self.trim_length:
            noisefile = noisefile[: self.trim_length]

        mixed = noisefile * factor * np.max(corr_data) / np.max(noisefile) + corr_data
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
