import argparse

def parse_args():
    parser = argparse.ArgumentParser("image_classification")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--use_gpu', type=bool, default=0, help='whether to use gpu')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of epoch')
    args = parser.parse_args()
    return args