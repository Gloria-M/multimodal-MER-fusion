"""
This module contains all the necessary functions to create data loaders.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_mfccs(sample_mfcc, mfcc_mean, mfcc_std):
    """
    Function to normalize MFCCs data, according to mean and variance in train set.
    :param sample_mfcc: MFCC data to be normalized
    :param mfcc_mean: train set MFCC mean
    :param mfcc_std: train set MFCC stadard deviation
    :return: normalized MFCC data
    """
    return (sample_mfcc - mfcc_mean) / mfcc_std


def load_annotations(data_dir, mode):
    """
    Function to load the annotations.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :return: annotations as tensors
    """
    annotations = np.load(os.path.join(data_dir, '{:s}_annotations.npy'.format(mode)))
    annotations = torch.tensor(annotations.astype(np.float32))

    return annotations

def load_audio_data(data_dir, mode):
    """
    Function to load the audio features.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :return: audio features as tensors
    """
    data = np.load(os.path.join(data_dir, '{:s}_mfccs.npy'.format(mode)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_text_data(data_dir, mode, modality):
    """
    Function to load the lyrics/comments tokens, according to the `modality`.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :param modality: select wether to load the lyrics or comments tokens
    :return: lyrics/comments tokens as tensors
    """
    data = np.load(os.path.join(data_dir, '{:s}_tokens_{:s}.npy'.format(mode, modality)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_embedding_matrix(data_dir, text_modality):
    """
    Function to load the embedding matrix with representations of lyrics/comments, according to `text_modality`.
    :param data_dir: path to data directory
    :param text_modality: select wether to load the lyrics or comments representations
    :return: embedding matrix as tensor
    """
    embedding_matrix = np.load(os.path.join(data_dir, 'embedding_matrix_{:s}.npy'.format(text_modality)))
    embedding_matrix = torch.tensor(embedding_matrix)

    return embedding_matrix


def make_loader(data_dir, mode, text_modality, mfcc_mean, mfcc_std, batch_size=64):
    """
    Function to create audio data loaders for training, validation or testing in batches with specified size.
    :param data_dir: path to data directory
    :param mode: type of data set to load - train | validation | test
    :param text_modality: the text modality to be combined with audio features (lyrics | comments)
    :param mfcc_mean: train set MFCC mean
    :param mfcc_std: train set MFCC stadard deviation
    :param batch_size: number of samples per training/testing batch
    :return: train/validation/test audio data loader
    """

    # Load audio & text data and annotations
    audio_data = load_audio_data(data_dir, mode)
    text_data = load_text_data(data_dir, mode, text_modality)
    annotations = load_annotations(data_dir, mode)

    # Create loader of combinations of data
    dataset_ = FusionDataset(audio_data, text_data, annotations, mfcc_mean, mfcc_std)
    dataloader_ = DataLoader(dataset_, batch_size=batch_size)

    return dataloader_


class FusionDataset(Dataset):
    """
    This class contains custom definition for creating fusion datasets, consisting in audio and text features.
    """
    def __init__(self, audio_data, text_data, annotations, mfcc_mean, mfcc_std):
        """
        :param audio_data: MFCCs data
        :param text_data: text features to be combined with audio features
        :param annotations: valence-arousal annotations
        :param mfcc_mean: train set MFCC mean
        :param mfcc_std: train set MFCC stadard deviation
        """
        self._audio_data = audio_data
        self._text_data = text_data
        self._annotations = annotations

        self._mfcc_mean = mfcc_mean
        self._mfcc_std = mfcc_std

    def __len__(self):

        return len(self._annotations)

    def __getitem__(self, idx):

        # Get audio and text data and annotations for sample at idx
        sample_mfcc = self._audio_data[idx]
        sample_mfcc = normalize_mfccs(sample_mfcc, self._mfcc_mean, self._mfcc_std)
        sample_text = self._text_data[idx]
        sample_annotations = self._annotations[idx]

        # Return audio and text features with the corresponding annotations
        return sample_mfcc, sample_text, sample_annotations
