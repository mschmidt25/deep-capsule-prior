import os

from dival.reconstructors.odl_reconstructors import FBPReconstructor

from dcp.reconstructors.dip import DeepImagePriorReconstructor
from dcp.reconstructors.dcp import DeepCapsulePriorReconstructor
from dcp.utils import Params
from dcp.utils.helper import load_standard_dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_reconstructor(method, dataset='ellipses', size_part=1.0,
                      pretrained=True, name=None):
    """Returns a reconstructor with hyperparameters/weights for the given
    dataset
    """
    if method == 'diptv':
        return diptv_reconstructor(
            dataset=dataset,
            name=name)
    
    if method == 'dcptv':
        return dcptv_reconstructor(
            dataset=dataset,
            name=name)
    
    return None


def diptv_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: The Deep Image Prior (DIP) + TV method for the specified dataset
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_diptv'.format(dataset))
        if name is None:
            name = 'DIP + TV'

        # fbp_params = Params.load('{}_fbp'.format(dataset))
        # fbp_reco = FBPReconstructor(standard_dataset.ray_trafo,
        #                             hyper_params=fbp_params.dict)

        reconstructor = DeepImagePriorReconstructor(
            ray_trafo=standard_dataset.ray_trafo,
            # ini_reco=fbp_reco,
            hyper_params=params.dict,
            name=name)

        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')
        
        
def dcptv_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: The Deep Capsule Prior (DCP) + TV method for the specified dataset
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_dcptv'.format(dataset))
        if name is None:
            name = 'DCP + TV'
    
        reconstructor = DeepCapsulePriorReconstructor(
            ray_trafo=standard_dataset.ray_trafo,
            hyper_params=params.dict,
            name=name)
    
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')
