import time
import logging
import sys
import os

BASE_DIR = './runs/' 
BASIC_FMT = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %M:%S")


def _get_logdir(base_dir):
    now = time.localtime()

    save_dir = os.path.join(BASE_DIR, base_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def _file_truncate(filename):
    if os.path.isfile(filename):
        p_file = open(filename, "w")
        p_file.truncate()
        p_file.close()

def setup_logger(base_dir):
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(BASIC_FMT)
    root_logger.addHandler(ch)

    save_dir = _get_logdir(base_dir)


    filename = os.path.join(save_dir, 'process.log')
    _file_truncate(filename)
   
    spr = logging.FileHandler(filename = filename)
    spr.setLevel(logging.INFO)
    spr.setFormatter(BASIC_FMT)
    root_logger.addHandler(spr)
    
    filename = os.path.join(save_dir, 'specific_process.log')
    _file_truncate(filename)
    
    pr = logging.FileHandler(filename = filename)
    pr.setLevel(logging.DEBUG)
    pr.setFormatter(BASIC_FMT)
    root_logger.addHandler(pr)

    return save_dir


def get_logger(name):

    logger = logging.getLogger(name)
    logger.info(f"{name} Logger Initialized")
    return logger