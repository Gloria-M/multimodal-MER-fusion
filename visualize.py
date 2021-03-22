import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import wget
from zipfile import ZipFile


class FontsScheme:
    def __init__(self, fonts_dir):

        font_url = "https://freefontsfamily.com/download/Times-New-Roman-Font/"
        font_zip_path = wget.download(font_url, out=fonts_dir)
        with ZipFile(font_zip_path, 'r') as zipObj:
            zipObj.extractall(path=fonts_dir)

        regular_font_path = os.path.join(fonts_dir, "Times New Roman/times new roman.ttf")
        bold_font_path = os.path.join(fonts_dir, "Times New Roman/times new roman bold.ttf")

        self.title_font = fm.FontProperties(fname=regular_font_path)
        self.title_font.set_size(14)
        self.title_font.set_style('normal')

        self.labels_font = fm.FontProperties(fname=regular_font_path)
        self.labels_font.set_size(12)
        self.labels_font.set_style('normal')

        self.text_font = fm.FontProperties(fname=regular_font_path)
        self.text_font.set_size(10)
        self.text_font.set_style('normal')


class ColorScheme:
    def __init__(self):
        self.low = '#0a8d7e'
        self.mid1 = '#97b3a5'
        self.mid2 = '#ff9b9b'
        self.high = '#c73427'


class GradientColorMap:
    def __init__(self, colors: list):
        self.colors = colors
        self.name = 'gradient_cmap'
        self.num_bins = 100

    def get_cmap(self):
        cmap = LinearSegmentedColormap.from_list(self.name, self.colors, self.num_bins)
        return cmap


class Visualizer:
    def __init__(self, fonts_dir, plots_dir):

        self._plots_dir = plots_dir
        self._fonts = FontsScheme(fonts_dir)
        self._colors = ColorScheme()
        self._cmap = GradientColorMap([self._colors.low, self._colors.mid1, self._colors.mid2, self._colors.high])

    def plot_loss(self, axis, train_loss, validation_loss, dimension):

        validation_plot, = axis.plot(validation_loss, color=self._colors.low, lw=1)
        train_plot, = axis.plot(train_loss, color=self._colors.high, lw=1)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        xlim = [0, axis.get_xlim()[1]]
        axis.set_xlim(xlim)
        axis.set_xticklabels([str(int(tick)) for tick in axis.get_xticks()],
                             fontproperties=self._fonts.labels_font)
        ylim = [.02, .1]
        axis.set_ylim(ylim)
        yticks = np.linspace(ylim[0], ylim[1], 6)
        axis.set_yticks(yticks)
        axis.set_yticklabels([str('{:.3f}'.format(tick)) for tick in yticks],
                             fontproperties=self._fonts.labels_font)

        axis.tick_params(axis='both', color='black', length=3, width=.75)

        legend = axis.legend([validation_plot, train_plot],
                             ['Test Loss', 'Train loss'],
                             prop=self._fonts.labels_font)

        axis.set_title('{:s} Loss'.format(dimension.upper()), fontproperties=self._fonts.title_font)

    def plot_losses(self, train_dict, validation_dict, text_modality):

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        self.plot_loss(ax[0], train_dict['valence_loss'], validation_dict['valence_loss'], 'Valence')
        self.plot_loss(ax[1], train_dict['arousal_loss'], validation_dict['arousal_loss'], 'Arousal')

        plt.savefig(os.path.join(self._plots_dir, 'audio_{:s}_loss.svg'.format(text_modality)), dpi=300)

    def plot_valence_residuals(self, axis, valence_dict):

        true_valence = valence_dict['true_annotations']
        pred_valence = valence_dict['pred_annotations']

        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(True)
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_aspect('equal')

        axes_limits = [-.01, 1.01]
        axis.set_xlim(axes_limits)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([0, 1], fontproperties=self._fonts.text_font)
        axis.set_ylim(axes_limits)
        axis.set_yticks([])
        axis.tick_params(axis='both', color='black', length=3, width=.75)

        y_coords = np.linspace(0, 1, len(true_valence), endpoint=True)
        sorted_idx = np.argsort(pred_valence)
        fit_t_valence = true_valence[sorted_idx]
        fit_p_valence = pred_valence[sorted_idx]

        for (v_val, v_coord) in zip(fit_t_valence, y_coords):
            true_v, = axis.plot(v_val, v_coord, color='white',
                                marker='.', markersize=5, markerfacecolor=self._colors.high,
                                markeredgecolor=self._colors.high)
        fit, = axis.plot(fit_p_valence, y_coords, color=self._colors.low, lw=2)

        axis.text(1.15, 0.515, 'Valence', fontproperties=self._fonts.labels_font,
                  horizontalalignment='right')
        legend = axis.legend([fit, true_v], ['Prediction', 'True'], prop=self._fonts.labels_font)

    def plot_valence_distances(self, axis, valence_dict):

        true_valence = valence_dict['true_annotations']
        pred_valence = valence_dict['pred_annotations']

        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(True)
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_aspect('equal')

        axes_limits = [-.01, 1.01]
        axis.set_xlim(axes_limits)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([0, 1], fontproperties=self._fonts.text_font)
        axis.set_ylim(axes_limits)
        axis.set_yticks([])
        axis.tick_params(axis='both', color='black', length=3, width=.75)

        sorted_idx = np.argsort(true_valence)
        fit_t_valence = true_valence[sorted_idx]
        fit_p_valence = pred_valence[sorted_idx]

        for (v_true, v_pred) in zip(fit_t_valence, fit_p_valence):
            pred_distances = axis.plot([v_pred, v_true],
                                       [v_true, v_true],
                                       lw=.5, color=self._colors.high)
            pred_v, = axis.plot(v_pred, v_true, color='white',
                                marker='.', markersize=6, markerfacecolor=self._colors.high,
                                markeredgecolor=self._colors.high)
        fit, = axis.plot(fit_t_valence, fit_t_valence, color=self._colors.low, lw=2)

        axis.text(1.15, 0.515, 'Valence', fontproperties=self._fonts.labels_font,
                  horizontalalignment='right')
        legend = axis.legend([fit, pred_v], ['True', 'Prediction'], prop=self._fonts.labels_font)

    def plot_valence_predictions(self, valence_dict, text_modality):

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        self.plot_valence_residuals(ax[0], valence_dict)
        self.plot_valence_distances(ax[1], valence_dict)

        plt.savefig(os.path.join(self._plots_dir, 'audio_{:s}_valence_predictions.svg'.format(text_modality)))

    def plot_arousal_residuals(self, axis, arousal_dict):

        true_arousal = arousal_dict['true_annotations']
        pred_arousal = arousal_dict['pred_annotations']

        axis.spines['left'].set_visible(True)
        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_aspect('equal')

        axes_limits = [-.01, 1.01]
        axis.set_xlim(axes_limits)
        axis.set_xticks([])
        axis.set_ylim(axes_limits)
        axis.set_yticks([0, 1])
        axis.set_yticklabels([0, 1], fontproperties=self._fonts.text_font)
        axis.tick_params(axis='both', color='black', length=3, width=.75)

        x_coords = np.linspace(0, 1, len(true_arousal), endpoint=True)
        sorted_idx = np.argsort(pred_arousal)
        fit_t_arousal = true_arousal[sorted_idx]
        fit_p_arousal = pred_arousal[sorted_idx]

        for (a_coord, a_val) in zip(x_coords, fit_t_arousal):
            true_a, = axis.plot(a_coord, a_val, color='white',
                                marker='.', markersize=5, markerfacecolor=self._colors.high,
                                markeredgecolor=self._colors.high)
        fit, = axis.plot(x_coords, fit_p_arousal, color=self._colors.low, lw=2)

        axis.text(0.515, 1.15, 'Arousal', fontproperties=self._fonts.labels_font,
                  verticalalignment='top', rotation=90)

        legend = axis.legend([fit, true_a], ['Prediction', 'True'], prop=self._fonts.labels_font)

    def plot_arousal_distances(self, axis, arousal_dict):

        true_arousal = arousal_dict['true_annotations']
        pred_arousal = arousal_dict['pred_annotations']

        axis.spines['left'].set_visible(True)
        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_aspect('equal')

        axes_limits = [-.01, 1.01]
        axis.set_xlim(axes_limits)
        axis.set_xticks([])
        axis.set_ylim(axes_limits)
        axis.set_yticks([0, 1])
        axis.set_yticklabels([0, 1], fontproperties=self._fonts.text_font)
        axis.tick_params(axis='both', color='black', length=3, width=.75)

        sorted_idx = np.argsort(true_arousal)
        fit_t_arousal = true_arousal[sorted_idx]
        fit_p_arousal = pred_arousal[sorted_idx]
        for (a_true, a_pred) in zip(fit_t_arousal, fit_p_arousal):
            pred_distances = axis.plot([a_true, a_true],
                                       [a_true, a_pred],
                                       lw=.5, color=self._colors.high)
            pred_a, = axis.plot(a_true, a_pred, color='white',
                                marker='.', markersize=6, markerfacecolor=self._colors.high,
                                markeredgecolor=self._colors.high)
        fit, = axis.plot(fit_t_arousal, fit_t_arousal, color=self._colors.low, lw=2)
        axis.text(0.515, 1.15, 'Arousal', fontproperties=self._fonts.labels_font,
                  verticalalignment='top', rotation=90)

        legend = axis.legend([fit, pred_a], ['True', 'Prediction'], prop=self._fonts.labels_font)

    def plot_arousal_predictions(self, arousal_dict, text_modality):

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        self.plot_arousal_residuals(ax[0], arousal_dict)
        self.plot_arousal_distances(ax[1], arousal_dict)

        plt.savefig(os.path.join(self._plots_dir, 'audio_{:s}_arousal_predictions.svg'.format(text_modality)))

    def plot_quadrant_predictions(self, valence_dict, arousal_dict, quadrants_dict, text_modality):

        fig, axis = plt.subplots(1, 1, figsize=(7, 7))

        true_annotations = [(v, a) for v, a in zip(valence_dict['true_annotations'], arousal_dict['true_annotations'])]
        pred_annotations = [(v, a) for v, a in zip(valence_dict['pred_annotations'], arousal_dict['pred_annotations'])]
        true_quadrant = quadrants_dict['true_annotations']

        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.set_aspect('equal')

        ticks = [0, 1]
        axes_limits = [-.01, 1.01]
        axis.set_xticks(ticks)
        axis.set_xticklabels(ticks, fontproperties=self._fonts.text_font)
        axis.set_xlim(axes_limits)
        axis.set_yticks(ticks)
        axis.set_yticklabels(ticks, fontproperties=self._fonts.text_font)
        axis.set_ylim(axes_limits)
        axis.tick_params(axis='both', color='black', length=3, width=.75)

        quadrant_colors = [self._colors.high, self._colors.high, self._colors.low, self._colors.low]
        for sample_idx in range(len(true_annotations)):

            x, y = pred_annotations[sample_idx]
            quadrant = true_quadrant[sample_idx] - 1
            if quadrant % 2 == 1:
                face_color = 'white'
            else:
                face_color = quadrant_colors[quadrant]
            axis.plot(x, y, marker='.', markersize=8, markerfacecolor=face_color,
                      markeredgecolor=quadrant_colors[quadrant])

        axis.text(1.01, 0.515, 'Valence', fontproperties=self._fonts.labels_font,
                  horizontalalignment='right')
        axis.text(0.515, 1.01, 'Arousal', fontproperties=self._fonts.labels_font,
                  verticalalignment='top', rotation=90)

        emotions_text = ['High-intensity\nPositive\n(Q1)', 'High-intensity\nNegative\n(Q2)',
                         'Low-intensity\nNegative\n(Q3)', 'Low-intensity\nPositive\n(Q4)']
        emotions_loc = [(1.01, 1.01), (0, 1.01), (0, 0), (1.01, 0)]
        emotions_align = [('right', 'top'), ('left', 'top'), ('left', 'bottom'), ('right', 'bottom')]
        legend_loc = [(.92, .93), (.08, .93), (.08, .015), (.92, .015)]
        legend_colors = [(self._colors.high, self._colors.high),
                         (self._colors.high, 'white'),
                         (self._colors.low, self._colors.low),
                         (self._colors.low, 'white')]
        for e in range(4):
            x, y = emotions_loc[e]
            halign, valign = emotions_align[e]
            axis.text(x, y, emotions_text[e], horizontalalignment=halign,
                      verticalalignment=valign, fontproperties=self._fonts.labels_font)
            x, y = legend_loc[e]
            edge_color, face_color = legend_colors[e]
            axis.plot(x, y, marker='.', markersize=10, markerfacecolor=face_color,
                      markeredgecolor=edge_color)

        axis.set_title('Quadrant Predictions', fontproperties=self._fonts.title_font)

        plt.savefig(os.path.join(self._plots_dir, 'audio_{:s}_quadrant_predictions.svg'.format(text_modality)))
