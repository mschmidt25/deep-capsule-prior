from warnings import warn
from tqdm import tqdm
import torch
from math import ceil
import os 
from copy import deepcopy
import json

import numpy as np
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from odl.contrib.torch import OperatorModule
from dival import StandardLearnedReconstructor
from dival.measure import PSNR

from dcp.utils.losses import poisson_loss, tv_loss
from dcp.utils.models import get_capsule_skip_model


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class CapsulePostprocessor(StandardLearnedReconstructor):
    HYPER_PARAMS = {
        'supervised':
            {'default': True,
             'choices': [False, True]},
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
        'same_filter':
            {'default': False,
             'choices': [False, True]},
        'use_bias':
            {'default': True,
             'choices': [False, True]},
        'weight_init':
            {'default': 'xavier_uniform',
             'choices': ['kaiming_normal', 'kaiming_uniform',
                         'xavier_normal', 'xavier_uniform']},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']}
            
    }
    """
    Deep Capsule Prior reconstructor
    """

    def __init__(self, ray_trafo, hyper_params=None, callback=None,
                 callback_func=None, callback_func_interval=100, 
                 log_dir=None, **kwargs):

        super().__init__(ray_trafo, **kwargs)

        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.domain_shape = ray_trafo.domain.shape
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval

        self.log_dir = log_dir

        self.batch_size = 16
        self.epochs = 20

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, dataset, start_from_checkpoint=None):
        
        supervised = self.hyper_params['supervised']
        lr = self.hyper_params['lr']
        loss_function = self.hyper_params['loss_function']
        gamma = self.hyper_params['gamma']

        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape))

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        # reset model before training
        self.init_model()
        if not start_from_checkpoint == None:
            self.load_learned_params(start_from_checkpoint)

        if not self.log_dir == None: 
            log_dir = self.log_dir
            created = False
            idx = 0
            while not created:
                log_dir = os.path.join(self.log_dir, "version_{}".format(idx))
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    self.log_dir = log_dir 
                    created = True
                idx = idx + 1
        else:
            log_dir = None

        if loss_function == 'mse':
            criterion = MSELoss()
        elif loss_function == 'poisson':
            criterion = poisson_loss
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()
            
        self.init_optimizer(dataset_train=dataset_train)

        data_loaders = {
            'train': DataLoader(
                dataset_train, batch_size=self.batch_size,
                num_workers=0, shuffle=False,
                pin_memory=False),
            'validation': DataLoader(
                dataset_validation, batch_size=self.batch_size,
                num_workers=0, shuffle=False,
                pin_memory=False)}

        dataset_sizes = {'train': len(dataset_train),
                         'validation': len(dataset_validation)}

        if not log_dir == None:
            writer = SummaryWriter(log_dir=self.log_dir, max_queue=10)

            with open(os.path.join(self.log_dir, "hparams.json"), 'w') as fp:
                json.dump(self.hyper_params, fp) 

        self.model.to(self.device)
        self.model.train()

        self.optimizer = Adam(self.model.parameters(), lr=lr)

        best_psnr = 0.

        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                
                with tqdm(data_loaders[phase],desc='epoch {:d}'.format(epoch + 1)) as pbar:
                    for inputs, labels in pbar:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()
            
                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)[1]
                            
                            if supervised:
                                loss = criterion(outputs, labels)
                            else:
                                loss = criterion(self.ray_trafo_module(outputs), inputs)
                            
                            loss = loss + gamma * tv_loss(outputs)
                            
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self.optimizer.step()

                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, labels_)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})

                        if self.log_dir is not None and phase == 'train':
                            step = (epoch * ceil(dataset_sizes['train']
                                                 / self.batch_size)
                                    + ceil(running_size / self.batch_size))
                            writer.add_scalar(
                                'loss/{}'.format(phase),
                                torch.tensor(running_loss/running_size), step)
                            writer.add_scalar(
                                'psnr/{}'.format(phase),
                                torch.tensor(running_psnr/running_size), step)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if self.log_dir is not None and phase == 'validation':
                        step = (epoch+1) * ceil(dataset_sizes['train']
                                                / self.batch_size)
                        writer.add_scalar('loss/{}'.format(phase),
                                          epoch_loss, step)
                        writer.add_scalar('psnr/{}'.format(phase),
                                          epoch_psnr, step)

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if not log_dir == None:
                            self.save_learned_params(
                                    self.log_dir + "dcptv_post_epoch={}.pt".format(epoch))
                    
                    if (phase == 'validation' and self.log_dir is not None):
                        with torch.no_grad():
                            reco = self.model(inputs)[1]

                            img_grid_gt = torchvision.utils.make_grid(labels,
                                            normalize=True, scale_each=True)

                            writer.add_image("ground_truth", img_grid_gt,
                                             epoch+1)

                            img_grid_reco = torchvision.utils.make_grid(reco,
                                            normalize=True, scale_each=True)
                            writer.add_image("reconstruction", img_grid_reco,
                                             epoch+1)

        print('Best val psnr: {:4f}'.format(best_psnr))

    def init_model(self):

        capsules = self.hyper_params['capsules']
        scales = self.hyper_params['scales']
        skip_capsules = self.hyper_params['skip_capsules']
        iter_rout = self.hyper_params['iter_rout']
        weight_init = self.hyper_params['weight_init']
        same_filter = self.hyper_params['same_filter']
        use_bias = self.hyper_params['use_bias']

        self.model = get_capsule_skip_model(
            in_ch=1,
            out_ch=1,
            capsules=capsules[:scales],
            skip_capsules=skip_capsules[:scales],
            iter_rout=iter_rout,
            weight_init=weight_init,
            same_filter=same_filter,
            use_bias=use_bias,
            op=self.ray_trafo)

    def _reconstruct(self, observation):
        if not torch.is_tensor(observation):
                observation = torch.from_numpy(
                    np.asarray(observation)[None, None]).to(self.device)
        x = self.model(observation)[1]

        return x

    def load_hyper_params(self, hparams_path):
        if not hparams_path.endswith(".json"):
            hparams_path = os.path.join(hparams_path, "hparams.json")

        with open(hparams_path, "r") as fp:
            data = json.load(fp)

        for key in data.keys():
            self.hyper_params[key] = data[key]


    def load_learned_params(self, path):
        """Load learned parameters from file.

        Parameters
        ----------
        path : str
            Path at which the learned parameters are stored.
            Implementations may interpret this as a file path or as a directory
            path for multiple files.
            If the implementation expects a file path, it should accept it
            without file ending.
        """
        path = path if path.endswith('.pt') else path + '.pt'
        self.init_model()
        map_location = (self.device if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)

        self.model.load_state_dict(state_dict)
      

    def get_layer_output(self, obs, layer_index):
        if layer_index < 0:  # -1 corresponds to input of last layer (outc)
            layer_index = self.model.scales * 2 + 1 + layer_index

        a, v = self.model.primary_capsule(obs)

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
