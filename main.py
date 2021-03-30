import os
import argparse

from utility_functions import *
from data_preprocessing import DataPreprocessor
from visualize import Visualizer
from train import Trainer
from test import Tester


def preprocess_data(args):
    """
    Function to process the data and create data sets.
    :param args: command line arguments
    """

    # Create object for preprocessing
    data_processor = DataPreprocessor(args)

    # Get data localization in quadrants
    data_processor.get_quadrants()
    # Augment dataset
    data_processor.augment_quadrants()
    # Create sets for train, validation and test
    data_processor.train_validation_test_split()

    # Extract features from audio, lyrics and comments
    data_processor.make_audio_datasets()
    data_processor.make_lyrics_comments_datasets()


def run_train(args):
    """
    Function to train a model.
    :param args: command line arguments
    """

    # Create directories for models and plots if they do not exist
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for training and visualization
    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    trainer = Trainer(args)

    for epoch in range(trainer.num_epochs):

        # Train and validate the model
        trainer.train()
        trainer.validate()

        # Display epoch every `log_interval`
        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.num_epochs:
            print_epoch(epoch + 1, trainer.train_dict, trainer.validation_dict)

        # Update the learing rate every `decay_interval`
        if (epoch + 1) % args.decay_interval == 0:
            trainer.update_learning_rate()

    # Visualize train and validation losses
    visualizer.plot_losses(trainer.train_dict, trainer.validation_dict, args.text_modality)
    # Save the trained model
    trainer.save_model()


def run_test(args):
    """
    Function for testing a model.
    :param args: command line arguments
    """

    # Create directories for plots if it doesn't exist
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for testing and visualization
    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    tester = Tester(args)

    # Make predictions on test set
    valence_dict, arousal_dict, quadrants_dict = tester.test()

    # Visualize valence predictions
    visualizer.plot_valence_predictions(valence_dict, args.text_modality)
    # Visualize arousal predictions
    visualizer.plot_arousal_predictions(arousal_dict, args.text_modality)
    # Visualize quadrant predictions
    visualizer.plot_quadrant_predictions(valence_dict, arousal_dict, quadrants_dict, args.text_modality)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Data', help="path to data directory")
    parser.add_argument('--audio_dir', type=str, default='./Data/Audio', help="path to audio data directory")
    parser.add_argument('--lyrics_dir', type=str, default='./Data/Lyrics', help="path to lyrics data directory")
    parser.add_argument('--comments_dir', type=str, default='./Data/Comments', help="path to comments data directory")

    parser.add_argument('--fonts_dir', type=str, default='./Fonts', help="path to font directory")
    parser.add_argument('--models_dir', type=str, default='./Models', help="path to models directory")
    parser.add_argument('--plots_dir', type=str, default='./Plots', help="path to plots directory")

    parser.add_argument('--samples_per_quadrant', type=int, default=3000,
                        help="number of samples in each of the four quadrants after augmentation")
    parser.add_argument('--train_val_test_ratio', type=float, nargs='+', default=[.7, .15, .15],
                        help="train, validation and test sets size. NOTE: must sum up to 1")

    parser.add_argument('--audio_exetnsion', type=str, default='mp3',
                        help="extension of audio samples in dataset")
    parser.add_argument('--mfcc_mean', type=float, default=-1.414, help="train MFCCs mean")
    parser.add_argument('--mfcc_std', type=float, default=52.411, help="train MFCCs standard deviation")

    parser.add_argument('--text_modality', type=str, default='lyrics',
                        help="the text modality to be combined with audio features")

    parser.add_argument('--device', type=str, default='cuda', help="use CUDA if available")
    parser.add_argument('--mode', type=str, default='train', help="train | test | preprocess - "
                                                                  "training / testing mode or to preprocess data")
    parser.add_argument('--load_pretrained', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--lr_init', type=float, default=1e-2)
    parser.add_argument('--lr_decay', type=float, default=1e-1)
    parser.add_argument('--decay_interval', type=int, default=1000, help="frequency of learning rate decrease")
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    args = parser.parse_args()

    print('\n\n')
    print_params(args.__dict__)
    print('\n\n')

    if args.mode == 'preprocess':
        preprocess_data(args)

    elif args.mode == 'train':
        run_train(args)

    elif args.mode == 'test':
        run_test(args)
