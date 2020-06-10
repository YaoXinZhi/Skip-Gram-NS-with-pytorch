# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 09/04/2020 下午3:20 
@Author: xinzhi yao 
"""

import os
import time
import importlib
import argparse
from tqdm import tqdm

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

import Skip_Gram


def init_config():
    parser = argparse.ArgumentParser(description='Skip-Gram')
    parser.add_argument('-ds', dest='dataset', required=True, help='Dataset name. ["rice", ]')
    args = parser.parse_args()


    config_file = 'config.config_{0}'.format(args.dataset)
    params = importlib.import_module(config_file.params)
    args = argparse.Namespace(**vars(args), **params)
    args.vocab = None
    args.vocab_size = None

    args.use_cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args

def main(args):

    global logging



