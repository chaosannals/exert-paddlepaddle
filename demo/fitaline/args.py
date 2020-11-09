import sys
import argparse

def parse_args():
    '''
    获取命令行参数
    '''

    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="启用 GPU")
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="number of epochs.")
    args = parser.parse_args()
    return args