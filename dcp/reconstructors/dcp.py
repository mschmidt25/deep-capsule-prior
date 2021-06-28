import os
from warnings import warn
from tqdm import tqdm
import torch
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    TENSORBOARD_AVAILABLE = False
else:
    TENSORBOARD_AVAILABLE = True
    from dival.measure import PSNR, SSIM

from odl.contrib.torch import OperatorModule
from odl.tomo import fbp_op
from dival import IterativeReconstructor

from dcp.utils.losses import poisson_loss, tv_loss
from dcp.utils.models import get_capsule_skip_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class DeepCapsulePriorReconstructor(IterativeReconstructor):
    HYPER_PARAMS = {
        'lr':
            {'default': 1e-3,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'scales':
            {'default': 4,
             'choices': [3, 4, 5, 6, 7]},
        'capsules':
            {'default': [[8, 2], [16, 2], [32, 2], [64, 2]]},
        'skip_capsules':
            {'default': [[4, 2], [4, 2], [4, 2], [4, 2]]},
        'iter_rout':
            {'default': 1},
        'use_fbp':
            {'default': True,
             'choices': [False, True]},
        'iter_input':
            {'default': False,
             'choices': [False, True]},
        'same_filter':
            {'default': False,
             'choices': [False, True]},
        'use_bias':
            {'default': True,
             'choices': [False, True]},
        'eps':
            {'default': 1e-6,
             'range': [1e-8, 1e-2]},
        'weight_init':
            {'default': 'xavier_uniform',
             'choices': ['kaiming_normal', 'kaiming_uniform',
                         'xavier_normal', 'xavier_uniform']},
        'iterations':
            {'default': 5000,
             'range': [1, 50000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']}
    }
    """
    Deep Capsule Prior reconstructor
    """

    def __init__(self, ray_trafo, hyper_params=None, callback=None,
                 callback_func=None, callback_func_interval=100,
                 log_dir=None, log_images=True, log_reco_interval=100,
                 log_best=False, log_gt=None, log_sub_dir_from_hp_fun=None,
                 log_hparams=False, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        log_dir : str, optional
            Path for writing Tensorboard logs.
            If `None` (the default), logging is disabled.
        log_images : bool, optional
            Whether to log images.
            Default: ``True``.
        log_reco_interval : int, optional
            Interval at which reconstruction images should be logged.
            The same interval is used for logging of PSNR and SSIM values when
            `log_gt` is specified.
            Default: ``100``.
        log_best : bool, optional
            Whether to use the best reconstruction (in terms of loss) rather
            than the current network output for logging of reconstructions and,
            if `log_gt` is specified, PSNR and SSIM values.
            Default: ``False``.
        log_gt : array, optional
            Ground truth image used for logging.
            If specified, PSNR and SSIM values are logged.
        log_sub_dir_from_hp_fun : callable, optional
            Callable receiving the argument `self.hyper_params` and returning
            the name of a sub-directory in `log_dir` to log into.
        log_hparams : bool, optional
            Whether to log hyper parameters with Tensorboard.
            Non-scalar values (like ``self.hyper_params['capsules']``) are
            converted to strings.
            Default: ``False``.
        """

        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, callback=callback, **kwargs)

        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.domain_shape = ray_trafo.domain.shape
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval
        self.log_dir = log_dir
        self.log_images = log_images
        self.log_reco_interval = log_reco_interval
        self.log_best = log_best
        self.log_gt = np.asarray(log_gt) if log_gt is not None else log_gt
        self.log_sub_dir_from_hp_fun = log_sub_dir_from_hp_fun
        self.log_hparams = log_hparams
        if self.log_dir is not None and not TENSORBOARD_AVAILABLE:
            raise RuntimeError(
                'Tensorboard is not available. Please either install a '
                'Tensorboard version supported by Pytorch or disable logging.')

        self.fbp_op = fbp_op(
            ray_trafo, frequency_scaling=0.7, filter_type='Hann')

    def get_layer_output(self, layer_index):
        if layer_index < 0:  # -1 corresponds to input of last layer (outc)
            layer_index = self.model.scales * 2 + 1 + layer_index

        a, v = self.model.primary_capsule(self.net_input)

        if layer_index == 0:
            return a, v

        xs = [[a, v]]
        for i in range(self.model.scales):
            a_temp, v_temp = self.model.down[i](xs[-1])
            xs.append([a_temp, v_temp])
            if layer_index == i+1:
                return xs[-1]
        a, v = xs[-1]

        for i in range(self.model.scales):
            a, v = self.model.up[i]([[a, v], xs[-2-i]])
            if layer_index == self.model.scales + 1 + i:
                return a, v

        # a: [?, C, F, F], v: [?, C, 1, 1, F, F]
        a, v = self.model.outc([a, v])
        return a, v

    def _reconstruct(self, observation, *args, **kwargs):
        torch.random.manual_seed(10)

        lr = self.hyper_params['lr']
        gamma = self.hyper_params['gamma']
        scales = self.hyper_params['scales']
        capsules = self.hyper_params['capsules']
        iterations = self.hyper_params['iterations']
        skip_capsules = self.hyper_params['skip_capsules']
        loss_function = self.hyper_params['loss_function']
        iter_rout = self.hyper_params['iter_rout']
        use_fbp = self.hyper_params['use_fbp']
        iter_input = self.hyper_params['iter_input']
        use_bias = self.hyper_params['use_bias']
        same_filter = self.hyper_params['same_filter']
        weight_init = self.hyper_params['weight_init']
        eps = self.hyper_params['eps']
        
        output_depth = 1
        input_depth = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_fbp:
            self.net_input = torch.from_numpy(self.fbp_op(observation).data)[None, None].to(device)
        else:
            self.net_input = 0.1 * \
                torch.randn(input_depth, *self.reco_space.shape)[None].to(device)
        self.model = get_capsule_skip_model(
            input_depth,
            output_depth,
            capsules=capsules[:scales],
            skip_capsules=skip_capsules[:scales],
            iter_rout=iter_rout,
            weight_init=weight_init,
            same_filter=same_filter,
            use_bias=use_bias,
            eps=eps).to(device)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

        y_delta = torch.tensor(observation.asarray(), dtype=torch.float32)
        y_delta = y_delta.view(1, 1, *y_delta.shape)
        y_delta = y_delta.to(device)

        if loss_function == 'mse':
            criterion = MSELoss()
        elif loss_function == 'poisson':
            criterion = poisson_loss
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        output = self.model(self.net_input)[1]
        best_loss = np.inf
        best_output = output.detach()
        #scheduler = StepLR(self.optimizer, 1000, 0.5)

        if self.log_dir is not None:
            log_dir = self.log_dir
            if self.log_sub_dir_from_hp_fun is not None:
                log_sub_dir = self.log_sub_dir_from_hp_fun(self.hyper_params)
                log_dir = os.path.join(log_dir, log_sub_dir)
            writer = SummaryWriter(log_dir=log_dir)
            if self.log_gt is not None and self.log_images:
                writer.add_image(
                    'ref_ground_truth', torch.from_numpy(self.log_gt), 0,
                    dataformats='HW')
            if self.log_hparams:
                hparams = self.hyper_params.copy()
                for k, v in hparams.items():
                    if not np.isscalar(v):
                        hparams[k] = str(v)
                writer.add_hparams(hparams, {'dummy_metric': 0})

        for i in tqdm(range(iterations+1)):  # last optimizer.step() is unused
            self.optimizer.zero_grad()
            output = self.model(self.net_input)[1]
            loss_discrepancy = criterion(
                self.ray_trafo_module(output), y_delta)
            loss_tv = tv_loss(output)
            loss = loss_discrepancy + gamma * loss_tv
            loss.backward()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = output.detach()

            if (i % self.callback_func_interval == 0 or i == iterations) and \
                self.callback_func is not None:
                self.callback_func(
                    iteration=i, reconstruction=best_output[0, 0, ...].cpu().numpy(),
                    loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(
                    best_output[0, 0, ...].cpu().numpy()))

            if self.log_dir is not None:
                writer.add_scalar('loss', loss, i)
                writer.add_scalar('loss_discrepancy', loss_discrepancy, i)
                writer.add_scalar('loss_tv', loss_tv, i)
                if i % self.log_reco_interval == 0 or i == iterations:
                    if self.log_images:
                        log_reco = (best_output if self.log_best else
                                    output.detach())
                        writer.add_image(
                            'reconstruction', log_reco, i, dataformats='NCHW')
                    if self.log_gt is not None:
                        log_reco_np = (best_output if self.log_best else
                                       output.detach()).cpu().numpy()[0, 0]
                        writer.add_scalar(
                            'psnr', PSNR(log_reco_np, self.log_gt), i)
                        writer.add_scalar(
                            'ssim', SSIM(log_reco_np, self.log_gt), i)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            #scheduler.step()
            
            if iter_input:
                self.net_input = output.detach()

        return self.reco_space.element(best_output[0, 0, ...].cpu().numpy())


