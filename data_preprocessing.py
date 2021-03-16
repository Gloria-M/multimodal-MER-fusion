import os
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
import librosa

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from gensim.models import Word2Vec

from utility_functions import *


def get_audio_mfccs(wave):
    sample_rate = 20050
    crop_space = 75 * sample_rate
    crop_idx = np.random.uniform(0, 1, 1)[0]
    win_length = int(30 * sample_rate / 1000)
    window_crops = np.linspace(0, 75, 15, endpoint=False) * sample_rate

    start_crop = int((len(wave) - crop_space) * crop_idx)
    end_crop = start_crop + crop_space
    sample = wave[start_crop:end_crop]

    windows = None
    for w in window_crops:
        p = int(w)
        if windows is None:
            windows = sample[p:p + 3 * sample_rate]
        else:
            windows = np.concatenate([windows, sample[p:p + 3 * sample_rate]])

    sample_mfcc = librosa.feature.mfcc(windows, sr=sample_rate, n_mfcc=20,
                                       n_fft=win_length, hop_length=win_length)

    return sample_mfcc


def augment_text(sample):

    augmentation_percent = .25
    max_augmentation = 250
    max_trials = 500

    sample_length = len(sample)

    augmented_count = max(int(sample_length * augmentation_percent), max_augmentation)
    trials_count = 0
    while augmented_count:
        word_idx = np.random.randint(0, sample_length)
        trials_count += 1
        word = sample[word_idx]
        synonym_sets = wordnet.synsets(word)

        if synonym_sets:
            if synonym_sets[0].lemmas()[0].name() != word:
                syn_word = synonym_sets[0].lemmas()[0].name()
                sample[word_idx] = syn_word
                augmented_count -= 1
            else:
                if len(synonym_sets[0].lemmas()) > 1:
                    syn_word = synonym_sets[0].lemmas()[1].name()
                    sample[word_idx] = syn_word
                    augmented_count -= 1
                elif len(synonym_sets) > 1:
                    syn_word = synonym_sets[1].lemmas()[0].name()
                    sample[word_idx] = syn_word
                    augmented_count -= 1

        if trials_count == max_trials:
            break
    return sample

def get_stemmed_text(stemmer, words_list):
    stems_list = [stemmer.stem(word) for word in words_list]

    return stems_list

def get_vocab_idxs(vocabulary):

    vocab_idxs = dict()
    for idx, word in enumerate(vocabulary):
        vocab_idxs[word] = idx

    return vocab_idxs

def convert_words_to_tokens(words_list, vocab_idxs):
    sequence_length = 500

    tokens_list = []
    for word in words_list:
        if word in vocab_idxs.keys():
            tokens_list.append(vocab_idxs[word])

        if len(tokens_list) == sequence_length:
            return tokens_list

    while len(tokens_list) < sequence_length:
        tokens_list.append(vocab_idxs['PAD'])

    return tokens_list

def make_embedding_matrix(w2v_model, w2v_vocab):
    vocab_size = len(w2v_vocab)
    num_features = w2v_model.wv.vector_size     # features = 300

    embedding_matrix = np.zeros((vocab_size, num_features))
    for idx, word in enumerate(w2v_vocab[:-1]):
        embedding_matrix[idx] = w2v_model.wv[word]

    return embedding_matrix


class DataPreprocessor:
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._audio_dir = args.audio_dir
        self._lyrics_dir = args.lyrics_dir
        self._comments_dir = args.comments_dir

        self._audio_extension = args.audio_extension

        with open(os.path.join(self._data_dir, 'annotations.json')) as infile:
            self._annotations_dict = json.load(infile)
        with open(os.path.join(self._lyrics_dir, 'lyrics.json')) as infile:
            self._lyrics_dict = json.load(infile)
        with open(os.path.join(self._comments_dir, 'comments.json')) as infile:
            self._comments_dict = json.load(infile)

        self._original_ids = np.array(list(self._annotations_dict.keys()))
        self._original_anotations = None
        self._original_quadrants = None

        self._desired_size = args.samples_per_quadrant
        self._train_val_test_ratio = np.array(args.train_val_test_ratio)

        self._augmented_ids = None
        self._augmented_annotations = None

        self.train_ids = None
        self.train_annotations = None
        self.validation_ids = None
        self.validation_annotations = None
        self.test_ids = None
        self.test_annotations = None

    def get_quadrants(self):

        annotations = []
        quadrants = []
        for track_id in self._original_ids:
            valence = self._annotations_dict[track_id]['valence']
            arousal = self._annotations_dict[track_id]['arousal']

            measurement = [valence, arousal]
            quadrant = get_quadrant(measurement)

            annotations.append(measurement)
            quadrants.append(quadrant)

        self._original_anotations = np.array(annotations)
        self._original_quadrants = np.array(quadrants)

    def augment_quadrants(self):

        quadrant_names = [1, 2, 3, 4]

        augmented_ids = []
        augmented_annotations = []
        for q_name in quadrant_names:

            q_idxs = np.where(self._original_quadrants == q_name)[0]
            q_size = len(q_idxs)
            print('\nQUADRANT {:d} : {:>4d} samples'.format(q_name, q_size))

            if q_size >= self._desired_size:
                q_augmented_idxs = q_idxs[np.array(random.sample(range(q_size), self._desired_size))]
                q_augmented_ids = self._original_ids[q_augmented_idxs]
                q_qugmented_annotations = self._original_anotations[q_augmented_idxs]

                print('    Choosing {:>4d} samples'.format(q_augmented_idxs))
                print('   Resulting {:>4d} samples'.format(len(q_augmented_ids)))

            else:
                augm_size = self._desired_size - q_size
                q_augmented_idxs = q_idxs[np.random.randint(q_size, size=augm_size)]
                q_augmented_ids = np.concatenate([self._original_ids[q_idxs], self._original_ids[q_augmented_idxs]])
                q_augmented_annotations = np.concatenate([self._original_anotations[q_idxs],
                                                          self._original_anotations[q_augmented_idxs]])

                print('     Keeping {:>4d} samples'.format(q_size))
                print('    Choosing {:>4d} samples'.format(q_augmented_idxs))
                print('   Resulting {:>4d} samples'.format(len(q_augmented_ids)))

            augmented_ids.extend(list(q_augmented_ids))
            augmented_annotations.extend(list(q_augmented_annotations))

        self._augmented_ids = np.array(augmented_ids)
        self._augmented_annotations = np.array(augmented_annotations)

    def train_validation_test_split(self):

        assert_msg = 'wrong values for parameter --train_validation_test_ratio : elements must sum up to 1'
        assert self._train_val_test_ratio.sum() == 1, fail_format(assert_msg)

        train_size = self._train_val_test_ratio[0]
        notrain_size = self._train_val_test_ratio[1] + self._train_val_test_ratio[2]

        self.train_ids, notrain_ids, self.train_annotations, notrain_annotations = train_test_split(
            self._augmented_ids, self._augmented_annotations, test_size=notrain_size)

        validation_size = self._train_val_test_ratio[1] / notrain_size
        self.test_ids, self.validation_ids, self.test_annotations, self.validation_annotations = train_test_split(
            notrain_ids, notrain_annotations, test_size=validation_size)

        np.save(os.path.join(self._data_dir, 'train_ids.npy'), self.train_ids)
        np.save(os.path.join(self._data_dir, 'validation_ids.npy'), self.train_ids)
        np.save(os.path.join(self._data_dir, 'test_ids.npy'), self.train_ids)

        np.save(os.path.join(self._data_dir, 'train_annotations.npy'), self.train_annotations)
        np.save(os.path.join(self._data_dir, 'validation_annotations.npy'), self.train_annotations)
        np.save(os.path.join(self._data_dir, 'test_annotations.npy'), self.train_annotations)

        data_sets = [self.train_annotations, self.validation_annotations, self.test_annotations]
        sets_names = ['TRAIN SET', 'VALIDATION SET', 'TEST SET']
        quadrant_names = [1, 2, 3, 4]
        for set_name, data_set in zip(sets_names, data_sets):
            print('{:s} : {:d} samples'.format(set_name, len(data_set)))
            quadrants = np.array([get_quadrant(measurement) for measurement in data_set])
            for q_name in quadrant_names:
                q_count = np.sum(quadrants == q_name)
                q_percentage = q_count / self._desired_size * 100
                print('   Quadrant {:d} : {:>4d} samples  -  {:.2f}%%'.format(q_name, q_count, q_percentage))

    def make_audio_datasets(self):

        ids_sets = [self.train_ids, self.validation_ids, self.test_ids]
        sets_names = ['train', 'validation', 'test']

        for name, ids in zip(sets_names, ids_sets):
            mfccs = []

            for sample_id in ids:
                audio_path = os.path.join(self._audio_dir, '{:s}.{:s}'.format(sample_id, self._audio_extension))
                wave, _ = librosa.load(audio_path, sr=20050)
                mfccs.append(get_audio_mfccs(wave))

            mfccs = np.array(mfccs)
            np.save(os.path.join(self._data_dir, '{:s}_mfccs.npy'.format(name)), mfccs)

    def make_lyrics_comments_datasets(self):

        ids_sets = [self.train_ids, self.validation_ids, self.test_ids]
        sets_names = ['train', 'validation', 'test']

        lyrics_stemmed = {name: [] for name in sets_names}
        comments_stemmed = {name: [] for name in sets_names}
        lyrics_corpus = []
        comments_corpus = []

        ps = PorterStemmer()

        used_ids = []
        for name, ids in zip(sets_names, ids_sets):

            for sample_id in ids:
                sample_lyrics = self._lyrics_dict[sample_id]
                sample_comments = self._comments_dict[sample_id]

                if sample_id in used_ids:
                    sample_lyrics = augment_text(sample_lyrics)
                    sample_comments = augment_text(sample_comments)

                lyrics_stemmed[name].append(get_stemmed_text(ps, sample_lyrics))
                lyrics_corpus.append(get_stemmed_text(ps, sample_lyrics))

                comments_stemmed[name].append(get_stemmed_text(ps, sample_comments))
                comments_corpus.append(get_stemmed_text(ps, sample_comments))

                used_ids.append(sample_id)

        lyrics_model = Word2Vec(lyrics_corpus, size=300, min_count=1, window=7, iter=10)
        lyrics_w2v_vocab = list(lyrics_model.wv.vocab.keys())
        lyrics_w2v_vocab.append('PAD')
        lyrics_vocab_idxs = get_vocab_idxs(lyrics_w2v_vocab)
        lyrics_embedding_matrix = make_embedding_matrix(lyrics_model, lyrics_w2v_vocab)
        np.save(os.path.join(self._data_dir, 'embedding_matrix_lyrics.npy'), lyrics_embedding_matrix)

        comments_model = Word2Vec(comments_corpus, size=300, min_count=1, window=7, iter=10)
        comments_w2v_vocab = list(comments_model.wv.vocab.keys())
        comments_w2v_vocab.append('PAD')
        comments_vocab_idxs = get_vocab_idxs(comments_w2v_vocab)
        comments_embedding_matrix = make_embedding_matrix(comments_model, comments_w2v_vocab)
        np.save(os.path.join(self._data_dir, 'embedding_matrix_comments.npy'), comments_embedding_matrix)

        for name, ids in zip(sets_names, ids_sets):

            lyrics_tokens = []
            comments_tokens = []
            for lyrics_stems, comments_stems in zip(lyrics_stemmed[name], comments_stemmed[name]):

                for lyrics in comments_stems:
                    lyrics_tokens.append(convert_words_to_tokens(lyrics, lyrics_vocab_idxs))

                for comments in lyrics_stems:
                    comments_tokens.append(convert_words_to_tokens(comments, comments_vocab_idxs))

                np.save(os.path.join(self._data_dir, '{:s}_tokens_lyrics.npy'.format(name)), lyrics_tokens)
                np.save(os.path.join(self._data_dir, '{:s}_tokens_comments.npy'.format(name)), comments_tokens)
