"""
This module contains the definitions of Neural Network models used for training on audio or text features.
"""
import torch
import torch.nn as nn


class AudioNet(nn.Module):
    """
    This class contains the definition of the architecture and the forward-pass of the model to be trained
     on audio features.
    """
    def __init__(self):
        super().__init__()

        in_ch = 20
        num_filters1 = 16
        num_filters2 = 16
        num_hidden = 34
        out_size = 2

        self._conv1 = nn.Sequential(nn.Conv1d(in_ch, num_filters1, 10, 1),
                                    nn.BatchNorm1d(num_filters1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(4, 4))
        self._conv2 = nn.Sequential(nn.Conv1d(num_filters1, num_filters2, 10, 1),
                                    nn.BatchNorm1d(num_filters2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(4, 4))
        self._pool = nn.AvgPool1d(4, 4)
        self._drop = nn.Dropout(0.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(22*16, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        x = self._drop(x)

        x = x.view(-1, 22*16)

        x = self._fc1(x)
        x = self._drop(x)
        x = self._act(x)
        x = self._fc2(x)

        return x


class TextNet(nn.Module):
    """
    This class contains the definition of the architecture and the forward-pass of the model to be trained
     on lyrics or comments features.
    """
    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, num_feats = embedding_matrix.shape
        in_ch = 1
        num_filters = 16
        num_hidden = 32
        out_size = 2

        # Load weights and freeze gradients for the embedding layer
        self._emb = nn.Embedding.from_pretrained(embedding_matrix)

        self._conv = nn.Sequential(nn.Conv2d(in_ch, num_filters, (10, num_feats), 1),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2, 1), 2),)
        self._pool = nn.AvgPool1d(4, 4)
        self._drop = nn.Dropout(0.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(16 * 61, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        x = self._emb(x)
        x = torch.unsqueeze(x, 1)

        x = x.float()
        x = self._conv(x)
        x = torch.squeeze(x)
        x = self._pool(x)

        x = x.view(-1, 16 * 61)
        x = self._drop(x)
        x = self._fc1(x)
        x = self._drop(x)
        x = self._fc2(x)

        return x


class FusionNet(nn.Module):
    """
    This class contains the definition of the architecture and the forward-pass of the fusion model to be trained
     on audio and lyrics or comments features.
    """
    def __init__(self, embedding_matrix):
        super().__init__()

        num_hidden = 256
        out_size = 2
        self._embedding_matrix = embedding_matrix

        # Initialize model for audio features
        self._model_audio = AudioNet()
        self._model_audio._fc1 = nn.Identity()
        self._model_audio._fc2 = nn.Identity()

        # Initialize model for text features
        self._model_text = TextNet(embedding_matrix)
        self._model_text._fc1 = nn.Identity()
        self._model_text._fc2 = nn.Identity()

        self._bn = nn.BatchNorm1d(1)
        self._drop = nn.Dropout(.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(22*16 + 16*61, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, audio_x, text_x):

        # Extract features from audio
        audio_x = self._model_audio(audio_x.clone())
        audio_x = audio_x.view(-1, 22*16)

        # Extract features from text
        text_x = self._model_text(text_x.clone())
        text_x = text_x.view(-1, 16*61)

        # Concatenate audio and text features extracted
        x = torch.cat((audio_x, text_x), dim=1)
        x = x.view(-1, 1, 22*16 + 61*16)

        x = self._bn(x)
        x = x.view(-1, 22*16 + 61*16)
        x = self._fc1(x)
        x = self._drop(x)
        x = self._act(x)
        x = self._fc2(x)

        return x

    def load_pretrained_audio_model(self, model_path):
        """
        Method to load pre-trained weights and freeze gradients for audio model.
        :param model_path: path to pre-trained audio model
        """
        self._model_audio = AudioNet()
        self._model_audio.load_state_dict(torch.load(model_path))
        for name, param in self._model_audio.named_parameters():
            param.requires_grad = False

        self._model_audio._fc1 = nn.Identity()
        self._model_audio._fc2 = nn.Identity()

    def load_pretrained_text_model(self, model_path):
        """
        Method to load pre-trained weights and freeze gradients for text model.
        :param model_path: path to pre-trained text model
        """
        self._model_text = TextNet(self._embedding_matrix)
        self._model_text.load_state_dict(torch.load(model_path))
        for name, param in self._model_text.named_parameters():
            param.requires_grad = False

        self._model_text._fc1 = nn.Identity()
        self._model_text._fc2 = nn.Identity()
