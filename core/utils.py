import os
import sys
import logging
import torch
import random
import numpy as np


def ensure_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def move_to_gpu(x):
    if torch.cuda.is_available():
        if isinstance(x, list):
            for k in range(len(x)):
                x[k] = x[k].cuda()
        else:
            x = x.cuda()
    return x


def setup_logger(log_file, level=logging.INFO, stdout=False):
    logger = logging.getLogger(log_file)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stdout is True:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    return logger


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
