import numpy as np
import os
import matplotlib.pyplot as plt
from dival.measure import PSNR, SSIM
import torch.nn.functional as F

from dcp.utils.helper import load_standard_dataset
from dcp.utils import Params
from dcp.reconstructors.dcp import DeepCapsulePriorReconstructor

from datetime import datetime


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 1)
obs, gt = test_data[0]

class DCPPlotActivationsCallbackFunc():
    def __init__(self, dcp_reconstructor, name, gt=None,
                 layer_indices=(-2, -1), channels=4, overlay=True,
                 overlay_color=(1., 0.8, 0., 0.6), vmax=None,
                 overlay_normalize=False, cbar=False,
                 save_figure_path=None, save_figure_formats=('png',),
                 save_figure_dpi=None, timestamp=True, iter_digits=6):
        self.dcp_reconstructor = dcp_reconstructor
        assert isinstance(self.dcp_reconstructor,
                          DeepCapsulePriorReconstructor)
        self.name = name
        self.gt = gt
        self.layer_indices = layer_indices
        self.channels = channels
        self.overlay = overlay
        self.overlay_color = overlay_color
        self.overlay_normalize = overlay_normalize
        self.vmax = vmax
        self.cbar = cbar
        if np.isscalar(self.channels):
            self.channels = (self.channels,) * len(self.layer_indices)
        self.save_figure_path = save_figure_path
        self.save_figure_formats = save_figure_formats
        if isinstance(self.save_figure_formats, str):
            self.save_figure_formats = [self.save_figure_formats]
        self.timestamp = timestamp
        if isinstance(self.timestamp, bool):
            self.timestamp = (datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
                              if self.timestamp else '')
        self.iter_digits = iter_digits

        if self.save_figure_path is not None:
            os.makedirs(self.save_figure_path, exist_ok=True)
        self.save_figure_dpi = save_figure_dpi

    def __call__(self, iteration, reconstruction, loss):
        basename = '_'.join(filter(None, [
            self.name,
            'act',
            self.timestamp,
            'iter{iteration:0{iter_digits}d}'.format(
                iteration=iteration, iter_digits=self.iter_digits)]))
        if self.save_figure_formats is not None:
            for fig_format in self.save_figure_formats:
                imshow_kwargs = {
                    'vmin': 0.,
                    'vmax': self.vmax,
                    'interpolation': 'none' if fig_format == 'pdf' else None,
                    'cmap': 'gray' if self.overlay else None
                }
                rows = 1 + len(self.layer_indices)
                columns = max(3, max(self.channels))
                fig, ax = plt.subplots(rows, columns,
                                       figsize=(3*columns, 3*rows))
                fig.subplots_adjust(hspace=0.3, top=0.9)
                fig.suptitle(
                    'DCP iter {iteration:0{iter_digits}d}'.format(
                        iteration=iteration, iter_digits=self.iter_digits))
                activation, _ = self.dcp_reconstructor.model(
                    self.dcp_reconstructor.net_input)
                activation_final = activation.detach().cpu().numpy()[0, 0]
                if self.overlay:
                    activation_final_overlay = np.zeros(
                        (4,) + activation_final.shape)
                    activation_final_overlay[:] = np.array(
                        self.overlay_color)[:, None, None]
                    activation_final_overlay[3] *= activation_final
                    if self.overlay_normalize:
                        activation_final_overlay[3] /= np.max(activation_final)
                    ax[0, 0].imshow(np.asarray(self.gt).T,
                                    **imshow_kwargs)
                    ax[0, 0].imshow(
                        np.asarray(activation_final_overlay).T,
                        **imshow_kwargs)
                else:
                    im = ax[0, 0].imshow(np.asarray(activation_final).T,
                                         **imshow_kwargs)
                    if self.cbar:
                        fig.colorbar(im, ax=ax[0, 0])
                ax[0, 0].set_title('Final activation')
                ax[0, 0].set_xticks([])
                ax[0, 0].set_yticks([])
                im = ax[0, 1].imshow(np.asarray(reconstruction).T,
                                     **imshow_kwargs)
                if self.cbar:
                    fig.colorbar(im, ax=ax[0, 1])
                ax[0, 1].set_title('Reconstruction')
                ax[0, 1].set_xticks([])
                ax[0, 1].set_yticks([])
                im = ax[0, 2].imshow(np.asarray(self.gt).T, **imshow_kwargs)
                if self.cbar:
                    fig.colorbar(im, ax=ax[0, 2])
                ax[0, 2].set_title('Ground truth')
                ax[0, 2].set_xticks([])
                ax[0, 2].set_yticks([])
                for j in range(2, columns):
                    ax[0, j].axis('off')
                for i, (l, c) in enumerate(zip(self.layer_indices,
                                               self.channels)):
                    activation, _ = self.dcp_reconstructor.get_layer_output(l)
                    activation = F.interpolate(activation, size=self.gt.shape,
                                               mode='bilinear')
                    c_ = min(c, activation.shape[1])
                    for j in range(c_):
                        activation_ch = activation.detach().cpu().numpy()[0, j]
                        if self.overlay:
                            activation_ch_overlay = np.zeros(
                                (4,) + activation_ch.shape)
                            activation_ch_overlay[:] = np.array(
                                self.overlay_color)[:, None, None]
                            activation_ch_overlay[3] *= activation_ch
                            if self.overlay_normalize:
                                activation_ch_overlay[3] /= np.max(
                                    activation_ch)
                            ax[i+1, j].imshow(np.asarray(self.gt).T,
                                            **imshow_kwargs)
                            im = ax[i+1, j].imshow(
                                np.asarray(activation_ch_overlay).T,
                                **imshow_kwargs)
                        else:
                            im = ax[i+1, j].imshow(np.asarray(activation_ch).T,
                                                   **imshow_kwargs)
                        if self.cbar:
                            fig.colorbar(im, ax=ax[i+1, j])
                        ax[i+1, j].set_xticks([])
                        ax[i+1, j].set_yticks([])
                        ax[i+1, j].set_title(
                            'Activation at layer {:d},\n'
                            'channel {:d}/{:d}'
                            .format(l, j, activation.shape[1]))
                    for j in range(c_, columns):
                        ax[i+1, j].axis('off')
                fig.savefig(
                    os.path.join(
                        self.save_figure_path, basename + '.' + fig_format),
                    bbox_inches='tight', dpi=self.save_figure_dpi)

def log_sub_dir_from_hp_fun(hyper_params):
    # ignore hyper parameters and pass timestamp
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

params = Params.load('ellipses_dcptv')
# params.dict['iterations'] = 100

# create the reconstructor
reconstructor = DeepCapsulePriorReconstructor(
    dataset.ray_trafo,
    hyper_params=params.dict,
    # log_dir='../utils/logs',
    # log_gt=gt,
    # log_sub_dir_from_hp_fun=log_sub_dir_from_hp_fun,
    # log_hparams=True,
    )

# visualize activations while iterating
reconstructor.callback_func = DCPPlotActivationsCallbackFunc(
    reconstructor, name='dcp', gt=gt, save_figure_path='../utils/figures')

reco = reconstructor.reconstruct(obs)

# visualize once by (ab-)using callback
# (useful for loading weights from file instead of calling `reconstruct`)
plot_activations = DCPPlotActivationsCallbackFunc(
    reconstructor, name='dcp', gt=gt, save_figure_path='../utils/figures')
plot_activations(reconstructor.hyper_params['iterations'],
                 reco, None)
