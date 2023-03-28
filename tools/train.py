import os
import sys
import argparse
import importlib
import cv2 as cv
import multiprocessing
import torch.nn as nn

def train(backbone, module_name):
    """RUNNING TRAIN SCRIPTS
    args:
        backbone: choose your backbone
        module_name:

    """


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('backbone', type=str, help='type of backbone network')
    parser.add_argument('module_name', type=str, help='name of module used')
    args = parser.parse_args()

    train(args.backbone, args.module_name)

if __name__ == '__main__':
    multiprocessing.set_start_method('spqwn', force=True)
    main()