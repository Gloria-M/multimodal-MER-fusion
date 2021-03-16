import os
import argparse

from utility_functions import *
from data_preprocessing import DataPreprocessor
from visualize import Visualizer
from train import Trainer
from test import Tester


def preprocess_data(args):

    data_processor = DataPreprocessor(args)

    data_processor.get_quadrants()
    data_processor.augment_quadrants()
    data_processor.train_validation_test_split()

    data_processor.make_audio_datasets()
    data_processor.make_lyrics_comments_datasets()


def run_train(args):

    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    trainer = Trainer(args)

    for epoch in range(trainer.num_epochs):

        trainer.train()
        trainer.validate()

        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.num_epochs:
            print_epoch(epoch + 1, trainer.train_dict, trainer.validation_dict)

        if (epoch + 1) % args.decay_interval == 0:
            trainer.update_learning_rate()

    visualizer.plot_losses(trainer.train_dict, trainer.validation_dict, args.text_modality)
    trainer.save_model()


def run_test(args):

    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    visualizer = Visualizer(args.fonts_dir, args.plots_dir)
    tester = Tester(args)

    valence_dict, arousal_dict, quadrants_dict = tester.test()

    visualizer.plot_valence_predictions(valence_dict, args.text_modality)
    visualizer.plot_arousal_predictions(arousal_dict, args.text_modality)
    visualizer.plot_quadrant_predictions(valence_dict, arousal_dict, quadrants_dict, args.text_modality)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--audio_dir', type=str, default='./Data/Audio')
    parser.add_argument('--lyrics_dir', type=str, default='./Data/Lyrics')
    parser.add_argument('--comments_dir', type=str, default='./Data/Comments')

    parser.add_argument('--samples_per_quadrant', type=int, default=3000)
    parser.add_argument('--train_val_test_ratio', type=float, nargs='+', default=[.7, .15, .15])

    parser.add_argument('--audio_exetnsion', type=str, default='mp3')
    parser.add_argument('--mfcc_mean', type=float, default=-1.414)
    parser.add_argument('--mfcc_std', type=float, default=52.411)

    parser.add_argument('--text_modality', type=str, default='lyrics')

    parser.add_argument('--models_dir', type=str, default='./Models')
    parser.add_argument('--plots_dir', type=str, default='./Plots')
    parser.add_argument('--fonts_dir', type=str, default='./Fonts')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_pretrained', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--lr_init', type=float, default=1e-2)
    parser.add_argument('--lr_decay', type=float, default=1e-1)
    parser.add_argument('--decay_interval', type=int, default=1000)
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
