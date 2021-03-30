"""
This module containg all the necessary methods for training the models to predict values for valence and arousal.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from models import FusionNet
from data_loader import make_loader, load_embedding_matrix
from utility_functions import *


class Trainer:
    """
    Methods for training are defined in this class.

    Attributes:
        train_loader, validation_loader: loading and batching the data in train and validation sets
        fusion_model: the fusion model to train
        train_dict, validation_dict: dictionaries with training information for
        computing perforformance and visualizing
    """
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        self._lr = args.lr_init
        self._lr_decay = args.lr_decay
        self._weight_decay = args.weight_decay

        self._text_modality = args.text_modality

        # Load the correct data according to `text_modality`
        self.train_loader = make_loader(self._data_dir, 'train', self._text_modality,
                                        args.mfcc_mean, args.mfcc_std)
        self.validation_loader = make_loader(self._data_dir, 'validation', self._text_modality,
                                             args.mfcc_mean, args.mfcc_std)

        self._embedding_matrix = load_embedding_matrix(self._data_dir, self._text_modality)

        self.fusion_model = FusionNet(self._embedding_matrix).to(self._device)
        if args.load_pretrained:
            self.load_model()

        self.optimizer = torch.optim.SGD(self.fusion_model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._criterion = nn.MSELoss()

        self.train_dict = {'valence_loss': [], 'arousal_loss': []}
        self.validation_dict = {'valence_loss': [], 'arousal_loss': []}

    def load_model(self):
        """
        Method to load the pretrained audio and text models used for feature extraction.
        """
        audio_model_path = os.path.join(self._models_dir, 'audio_model.pt')
        self.fusion_model.load_pretrained_audio_model(audio_model_path)

        text_model_path = os.path.join(self._models_dir, '{:s}_model.pt'.format(self._text_modality))
        self.fusion_model.load_pretrained_text_model(text_model_path)

    def save_model(self):
        """
        Method to save the trained model weights to a specified path.
        """
        model_path = os.path.join(self._models_dir, 'audio_{:s}_model.pt'.format(self._text_modality))
        torch.save(self.fusion_model.state_dict(), model_path)

    def update_learning_rate(self):
        """
        Method to update the learning rate, according to a decay factor.
        """
        self._lr *= self._lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr

        success_message = 'Learning rate updated to {:.1e}'.format(self._lr)
        print(success_format(success_message))

    def train(self):
        """
        Method to train models.
        """
        true_annotations = []
        pred_annotations = []

        self.fusion_model.train()
        # Iterate over the train set
        for batch_idx, (audio_data, text_data, annotations) in enumerate(self.train_loader):

            # Convert the data to the correct type and move to device
            audio_data = audio_data.to(self._device)
            text_data = text_data.long()
            text_data = text_data.to(self._device)
            annotations = annotations.to(self._device)

            # Zero-out the gradients and make predictions
            self.optimizer.zero_grad()
            output = self.fusion_model(audio_data, text_data)

            true_annotations.extend(annotations.cpu().detach().numpy())
            pred_annotations.extend(output.cpu().detach().numpy())

            # Compute the batch loss and gradients
            batch_loss = self._criterion(output, annotations)
            batch_loss.backward()

            # Update the weights
            self.optimizer.step()

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        # Extract predictions and true values for valence dimension and compute MSE
        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Extract predictions and true values for arousal dimension and compute MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.train_dict['valence_loss'].append(valence_mse)
        self.train_dict['arousal_loss'].append(arousal_mse)

    def validate(self):
        """
        Method to validate the models.
        """
        true_annotations = []
        pred_annotations = []

        self.fusion_model.eval()
        # Freeze gradients
        with torch.no_grad():
            # Iterate over validation set
            for batch_idx, (audio_data, text_data, annotations) in enumerate(self.validation_loader):

                # Convert the data to the correct type and move to device
                audio_data = audio_data.to(self._device)
                text_data = text_data.long()
                text_data = text_data.to(self._device)
                annotations = annotations.to(self._device)

                # Make predictions
                output = self.fusion_model(audio_data, text_data)

                true_annotations.extend(annotations.cpu().detach().numpy())
                pred_annotations.extend(output.cpu().detach().numpy())

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        # Extract predictions and true values for valence dimension and compute MSE
        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Extract predictions and true values for arousal dimension and compute MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.validation_dict['valence_loss'].append(valence_mse)
        self.validation_dict['arousal_loss'].append(arousal_mse)
