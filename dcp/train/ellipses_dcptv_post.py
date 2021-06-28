from dival import TaskTable
from dival.measure import PSNR, SSIM
from dival.datasets import get_standard_dataset

from dcp.utils import Params
from dcp.utils.helper import select_hyper_best_parameters
from dcp.utils.reports import save_results_table
from dcp.reconstructors.dcp_post import CapsulePostprocessor


# load data
dataset = get_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 5)

task_table = TaskTable()

# create the reconstructor
reconstructor = CapsulePostprocessor(dataset.ray_trafo,
                            log_dir='../utils/logs/ellipses/dcptv_post')
reconstructor.batch_size = 16
reconstructor.epochs = 1

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor,
                  dataset=dataset,
                  measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'supervised': [False],
                      'lr': [0.1],
                      'scales': [5],
                      'gamma': [1.e-03],
                      'weight_init': ['xavier_uniform'],
                      'same_filter': [False],
                      'use_bias': [True],
                      'iter_rout': [1,2,3],
                      'capsules': [[[1, 32],[2, 64],[2, 64],[2, 128],[2, 128]]],
                      'skip_capsules': [[[2,64], [2,64], [2,128], [2,128], [2,128]]]
                  })

results = task_table.run()

save_results_table(results, 'ellipses_dcptv_post')

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('ellipses_dcptv_post')
