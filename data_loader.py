import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_mfccs(sample_mfcc, mfcc_mean, mfcc_std):

    return (sample_mfcc - mfcc_mean) / mfcc_std


class FusionDataset(Dataset):

    def __init__(self, audio_data, text_data, annotations, mfcc_mean, mfcc_std):

        self._audio_data = audio_data
        self._text_data = text_data
        self._annotations = annotations

        self._mfcc_mean = mfcc_mean
        self._mfcc_std = mfcc_std

    def __len__(self):

        return len(self._annotations)

    def __getitem__(self, idx):

        sample_mfcc = self._audio_data[idx]
        sample_mfcc = normalize_mfccs(sample_mfcc, self._mfcc_mean, self._mfcc_std)
        sample_text = self._text_data[idx]
        sample_annotations = self._annotations[idx]

        return sample_mfcc, sample_text, sample_annotations


def load_annotations(data_dir, mode):
    annotations = np.load(os.path.join(data_dir, '{:s}_annotations.npy'.format(mode)))
    annotations = torch.tensor(annotations.astype(np.float32))

    return annotations

def load_audio_data(data_dir, mode):
    data = np.load(os.path.join(data_dir, '{:s}_mfccs.npy'.format(mode)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_text_data(data_dir, mode, modality):
    data = np.load(os.path.join(data_dir, '{:s}_tokens_{:s}.npy'.format(mode, modality)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_embedding_matrix(data_dir, text_modality):

    embedding_matrix = np.load(os.path.join(data_dir, 'embedding_matrix_{:s}.npy'.format(text_modality)))
    embedding_matrix = torch.tensor(embedding_matrix)

    return embedding_matrix


def make_loader(data_dir, mode, text_modality, mfcc_mean, mfcc_std, batch_size=64):

    audio_data = load_audio_data(data_dir, mode)
    text_data = load_text_data(data_dir, mode, text_modality)
    annotations = load_annotations(data_dir, mode)

    dataset_ = FusionDataset(audio_data, text_data, annotations, mfcc_mean, mfcc_std)
    dataloader_ = DataLoader(dataset_, batch_size=batch_size)

    return dataloader_
