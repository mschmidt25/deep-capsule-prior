import argparse
from dival import DataPairs

from dcp.reconstructors import get_reconstructor
from dcp.utils.helper import load_standard_dataset
from dcp.utils.plot import plot_reconstructors_tests
from dcp.utils.helper import set_use_latex


set_use_latex()

def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='dcptv')
    parser.add_argument('--dataset', type=str, default='lodopab')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=100)
    return parser


def main():
    options = get_parser().parse_args()
    # load data
    dataset = load_standard_dataset(options.dataset, ordered=True)
    test_data = dataset.get_data_pairs('test', 100)

    sizes = [1.00]
    reconstructors = []

    for size_part in sizes:
        reconstructors.append(get_reconstructor(options.method,
                                                dataset=options.dataset,
                                                size_part=size_part,
                                                pretrained=True))

    for i in range(options.start, options.count):
        obs, gt = test_data[i]
        test_data = DataPairs([obs], [gt], name='test')

        # compute and plot reconstructions
        plot_reconstructors_tests(reconstructors, dataset.ray_trafo, test_data,
                                  save_name='{}-{}-test-{}'.format(
                                      options.dataset, options.method, i),
                                  fig_size=(9, 3),
                                  cmap='bone')


if __name__ == "__main__":
    main()
