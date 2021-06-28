import torch.nn as nn
from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule

class FBPModule(nn.Module):
    """
    Torch module of the filtered back-projection (FBP).

    Methods
    -------
    forward(x)
        Compute the FBP.
        
    """
    
    def __init__(self, ray_trafo, filter_type='Hann', frequency_scaling=1.):
        super().__init__()
        self.ray_trafo = ray_trafo
        
        filter_op = fbp_filter_op(self.ray_trafo, filter_type=filter_type,
                          frequency_scaling=frequency_scaling)
        self.filter_mod = OperatorModule(filter_op)
        self.ray_trafo_adjoint_mod = (
            TorchRayTrafoParallel2DAdjointModule(self.ray_trafo))
        
    def forward(self, x):
        x = self.filter_mod(x)
        x = self.ray_trafo_adjoint_mod(x)
        return x
