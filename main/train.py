import os
import sys
import argparse
import importlib
import cv2 as cv
import multiprocessing
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

import tools.settings as st


def train(train_module, train_name, cudnn_benchmark=True):
    """RUNNING TRAIN SCRIPTS
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = st.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = '{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, default='AE', help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default='autoencoder', help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0).')

    args = parser.parse_args()

    train(args.train_module, args.train_name, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
