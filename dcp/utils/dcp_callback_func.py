# -*- coding: utf-8 -*-
import os
from datetime import datetime
import torch
import numpy as np
from dcp.reconstructors import DeepCapsulePriorReconstructor
from dival.util.plot import plot_images
from dival.measure import PSNR, SSIM

class DCPCallbackFunc():
    def __init__(self, dcp_reconstructor, name, gt=None,
                 save_weights_path=None, save_reco_path=None,
                 save_figure_path=None, save_figure_formats=('png',),
                 save_figure_dpi=None, save_figure_exclude_gt=False,
                 timestamp=True, iter_digits=6):
        self.dcp_reconstructor = dcp_reconstructor
        assert isinstance(self.dcp_reconstructor,
                          DeepCapsulePriorReconstructor)
        self.name = name
        self.gt = gt
        self.save_weights_path = save_weights_path
        self.save_reco_path = save_reco_path
        self.save_figure_path = save_figure_path
        self.save_figure_formats = save_figure_formats
        if isinstance(self.save_figure_formats, str):
            self.save_figure_formats = [self.save_figure_formats]
        self.save_figure_exclude_gt = save_figure_exclude_gt
        self.timestamp = timestamp
        if isinstance(self.timestamp, bool):
            self.timestamp = (datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
                              if self.timestamp else '')
        self.iter_digits = iter_digits

        if self.save_weights_path is not None:
            os.makedirs(self.save_weights_path, exist_ok=True)
        if self.save_reco_path is not None:
            os.makedirs(self.save_reco_path, exist_ok=True)
        if self.save_figure_path is not None:
            os.makedirs(self.save_figure_path, exist_ok=True)
        self.save_figure_dpi = save_figure_dpi

    def __call__(self, iteration, reconstruction, loss):
        basename = '_'.join(filter(None, [
            self.name,
            self.timestamp,
            'iter{iteration:0{iter_digits}d}'.format(
                iteration=iteration, iter_digits=self.iter_digits)]))
        if self.save_weights_path is not None:
            torch.save(
                self.dcp_reconstructor.model.state_dict(),
                os.path.join(self.save_weights_path, basename + '.pt'))
        if self.save_reco_path is not None:
            np.save(
                os.path.join(self.save_reco_path, basename + '.npy'),
                reconstruction)
        if self.save_figure_formats is not None:
            for fig_format in self.save_figure_formats:
                interpolation = 'none' if fig_format == 'pdf' else None
                fig_size = (7, 6)
                images = [reconstruction]
                if self.gt is not None and not self.save_figure_exclude_gt:
                    fig_size = (15, 6)
                    images += [self.gt]
                im, ax = plot_images(
                    images,
                    interpolation=interpolation,
                    fig_size=fig_size)
                ax[0].set_title(
                    'DCP iter {iteration:0{iter_digits}d}'.format(
                        iteration=iteration, iter_digits=self.iter_digits))
                if self.gt is not None and not self.save_figure_exclude_gt:
                    ax[1].set_title('Ground truth')
                xlabel = 'loss: {:f}'.format(loss)
                if self.gt is not None:
                    psnr = PSNR(reconstruction, self.gt)
                    ssim = SSIM(reconstruction, self.gt)
                    xlabel = (
                        'PSNR: {:.2f} dB, SSIM: {:.3f}\n'.format(psnr, ssim)
                        + xlabel)
                ax[0].set_xlabel(xlabel)
                ax[0].figure.savefig(
                    os.path.join(
                        self.save_figure_path, basename + '.' + fig_format),
                    bbox_inches='tight', dpi=self.save_figure_dpi)
