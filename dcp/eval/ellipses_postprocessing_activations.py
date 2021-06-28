import numpy as np
import os
import matplotlib.pyplot as plt
from dival.measure import PSNR, SSIM
import torch.nn.functional as F
import torch 
from torch.utils.data import DataLoader

from dcp.utils.helper import load_standard_dataset
from dival.datasets import get_standard_dataset
from dival.datasets.cached_dataset import CachedDataset

from dcp.utils import Params
from dcp.reconstructors.capnet import CapsulePostprocessor

from datetime import datetime

# load data

#dataset = get_standard_dataset("lodopab", impl="astra_cpu")
#cachepath = "/localdata/AlexanderDenker/dival_dataset_caches"
#cachefiles = {'test': (os.path.join(cachepath, "cache_test_lodopab_fbp.npy" ),
#                             os.path.join(cachepath, "cache_test_lodopab_gts.npy"))}

dataset = get_standard_dataset("ellipses", impl="astra_cpu")
cachepath = "/localdata/dival_dataset_caches"
cachefiles = {'test': (os.path.join(cachepath, "cache_test_ellipses_fbp.npy" ),
                             os.path.join(cachepath, "cache_test_ellipses_gts.npy"))}

cached_dataset = CachedDataset(dataset, (dataset.space[1], dataset.space[1]), cachefiles)

dataset_test = cached_dataset.create_torch_dataset(
            part='test', reshape=((1,) + dataset.space[1].shape,
                                        (1,) + dataset.space[1].shape))
test_loader = DataLoader(
                dataset_test, batch_size=1,
                num_workers=0, shuffle=True,
                pin_memory=False)
obs, gt = next(iter(test_loader))

print(obs.shape, gt.shape)

class DCPPlotActivationsCallbackFunc():
    def __init__(self, dcp_reconstructor, name, gt=None,
                 layer_indices=(-2, -1), channels=4, overlay=True,
                 overlay_color=(1., 0.8, 0., 0.6), vmax=None,
                 overlay_normalize=False, cbar=False,
                 save_figure_path=None, save_figure_formats=('png',),
                 save_figure_dpi=None, timestamp=True, iter_digits=6):
        self.dcp_reconstructor = dcp_reconstructor
        #assert isinstance(self.dcp_reconstructor,
        #                  DeepCapsulePriorReconstructor)
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

    def __call__(self, observation, reconstruction, loss):
        basename = '_'.join(filter(None, [
            self.name,
            'act',
            self.timestamp]))
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
                    'CapNet')
                activation, _ = self.dcp_reconstructor.model(observation)

                activation_final = activation.detach().cpu().numpy()[0, 0]

                if self.overlay:
                    activation_final_overlay = np.zeros(
                        (4,) + activation_final.shape)
                    activation_final_overlay[:] = np.array(
                        self.overlay_color)[:, None, None]
                    activation_final_overlay[3] *= activation_final
                    if self.overlay_normalize:
                        activation_final_overlay[3] /= np.max(activation_final)
                    ax[0, 0].imshow(np.asarray(self.gt.squeeze()).T,
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
                im = ax[0, 1].imshow(np.asarray(reconstruction.cpu().squeeze()).T,
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
                    activation, _ = self.dcp_reconstructor.get_layer_output(obs, l)
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

# params.dict['iterations'] = 100

# create the reconstructor


experiment_name = "ellipses"
version = "version_3"
path_parts = ['..', 'experiments', experiment_name, version]
log_dir = os.path.join(*path_parts)
checkpoint = "caps_net_epoch=19.pt"

cap_net = CapsulePostprocessor(dataset.ray_trafo)
cap_net.device = 'cpu'#"cuda:5"
cap_net.load_hyper_params(log_dir)
print(cap_net.hyper_params)
cap_net.init_model()
cap_net.load_learned_params(os.path.join(log_dir, checkpoint))
cap_net.model.to(cap_net.device)
cap_net.model.eval()

# visualize activations while iterating
#reconstructor.callback_func = DCPPlotActivationsCallbackFunc(
#    reconstructor, name='dcp', gt=gt, save_figure_path='../utils/figures')

with torch.no_grad():
    obs = obs.to(cap_net.device)
    reco = cap_net._reconstruct(obs)
"""
#print(reco.shape)
#plt.figure()
#plt.imshow(reco.numpy()[0,0,:,:])
#plt.show()
# visualize once by (ab-)using callback
# (useful for loading weights from file instead of calling `reconstruct`)
plot_activations = DCPPlotActivationsCallbackFunc(
    cap_net, name='dcp', gt=gt.cpu().squeeze(), 
    save_figure_path='../utils/figures', layer_indices=(5,6,7,8,9), channels=8, overlay=True,overlay_color=(1., 0.8, 0., 0.85),
    overlay_normalize=False)

plot_activations(obs,reco, None)


plot_activations = DCPPlotActivationsCallbackFunc(
    cap_net, name='dcp', gt=gt.cpu().squeeze(), 
    save_figure_path='../utils/figures', layer_indices=(5,6,7,8,9), channels=8, overlay=False,overlay_color=(1., 0.8, 0., 0.85),
    overlay_normalize=False)

plot_activations(obs,reco, None)
"""