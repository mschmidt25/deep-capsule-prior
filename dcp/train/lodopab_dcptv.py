import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dcp.utils import Params
from dcp.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dcp.reconstructors.dcp import DeepCapsulePriorReconstructor
from dcp.utils.reports import save_results_table

from dcp.utils.dcp_callback_func import DCPCallbackFunc
from datetime import datetime

# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('validation', 5)
obs, gt = test_data[0]

task_table = TaskTable()

def log_sub_dir_from_hp_fun(hyper_params):
    # ignore hyper parameters and pass timestamp
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

# create the reconstructor
reconstructor = DeepCapsulePriorReconstructor(
    dataset.ray_trafo, log_dir='../utils/logs/lodopab/dcptv', log_gt=gt,
    log_sub_dir_from_hp_fun=log_sub_dir_from_hp_fun, log_hparams=True)
reconstructor.callback_func = DCPCallbackFunc(
    reconstructor, name='dcptv', gt=gt,
    save_weights_path='../utils/weights/lodopab/dcptv',
    save_reco_path='../utils/recos/lodopab/dcptv',
    save_figure_path='../utils/figures/lodopab/dcptv')

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr': [0.01],
                      'scales': [6],
                      'gamma': [16],
                      'use_fbp': [False, True],
                      'iter_input': [False],
                      'weight_init': ['xavier_uniform'],
                      'same_filter': [False],
                      'use_bias': [True],
                      'iter_rout': [1, 2],
                      'loss_function': ['poisson'],
                      'capsules': [[[1,16], [1,32], [2,64],
                                    [2,64], [2,128], [2,128]]],
                      'skip_capsules': [[[1, 32], [2, 64], [2, 64],
                                         [2, 128], [2, 128], [2, 128]]],
                      'iterations': [3000, 4000, 5000, 6000, 7000, 8000, 9000,
                                     10000, 11000, 12000, 13000, 14000, 15000,
                                     16000, 17000, 18000, 19000, 20000]
                  })

results = task_table.run()

save_results_table(results, 'lodopab_dcptv')

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('lodopab_dcptv')
