"""
This module containg all the necessary methods for testing the models trained to predict values
for valence and arousal.
"""
import os
import numpy as np
import torch

from models import FusionNet
from data_loader import make_loader, load_embedding_matrix
from utility_functions import *


class Tester:
    """
    Methods for testing are defined in this class.

    Attributes:
        test_loader: loading and batching the data in test set
        fusion_model: the fusion model to test
        train_dict, validation_dict: dictionaries with training information for
        computing perforformance and visualizing
    """
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir

        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self._text_modality = args.text_modality
        self.test_loader = make_loader(self._data_dir, 'test', self._text_modality,
                                       args.mfcc_mean, args.mfcc_std)

        self._embedding_matrix = load_embedding_matrix(self._data_dir, self._text_modality)
        self.fusion_model = self.load_model()

    def load_model(self):
        """
        Method to load the pretrained model.
        """
        fusion_model = FusionNet(self._embedding_matrix).to(self._device)
        model_path = os.path.join(self._models_dir, 'audio_{:s}_model.pt'.format(self._text_modality))
        fusion_model.load_state_dict(torch.load(model_path))

        return fusion_model

    def test(self):
        """
        Method to test the model.
        """
        true_annotations = []
        pred_annotations = []

        self.fusion_model.eval()
        # Freeze gradients
        with torch.no_grad():
            # Iterate over test set
            for batch_idx, (audio_data, text_data, annotations) in enumerate(self.test_loader):

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
        valence_mae = np.mean(np.abs(true_valence - pred_valence))
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        valence_dict = {'true_annotations': true_valence,
                        'pred_annotations': pred_valence,
                        'mae': valence_mae,
                        'mse': valence_mse}

        # Extract predictions and true values for arousal dimension and compute MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mae = np.mean(np.abs(true_arousal - pred_arousal))
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        arousal_dict = {'true_annotations': true_arousal,
                        'pred_annotations': pred_arousal,
                        'mae': arousal_mae,
                        'mse': arousal_mse}

        # Extract information about quadrants from valence & arousal annotations
        true_quadrant = np.array([get_quadrant(measurement) for measurement in true_annotations])
        pred_quadrant = np.array([get_quadrant(measurement) for measurement in pred_annotations])

        quadrants_dict = {'true_annotations': true_quadrant,
                          'pred_annotations': pred_quadrant}

        quadrants_names = [1, 2, 3, 4]
        # Compute quadrants accuracy
        for quadrant in quadrants_names:

            q_pred = true_quadrant[np.where(pred_quadrant == quadrant)]
            q_true = true_quadrant[np.where(true_quadrant == quadrant)]
            q_perc = np.sum(q_pred == quadrant) / len(q_true) * 100
            quadrants_dict[quadrant] = q_perc

        return valence_dict, arousal_dict, quadrants_dict
