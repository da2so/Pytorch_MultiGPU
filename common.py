import os
import inspect
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.distributed as dist

import dataset
import networks
from log import get_logger

logger = get_logger(__name__)


TIME_DICT = {}

def get_time_list():
    global TIME_DICT
    time_list = []
    for key, val_dict in TIME_DICT.items():

        if val_dict['count'] >= 2 : 
            avg_key = f'avg_{key}'
            total_key = f'total_{key}'

            avg_time = val_dict['time'] / val_dict['count']
            total_time = val_dict['time']
            time_list += [(avg_key, avg_time)]
            time_list += [(total_key, total_time)]
        else:
            time_list += [(key, val_dict['time'])]

    return time_list


def elapse_time(func): 
    """
    calculate the elapse time for each function (decorator)
    """
    def deco(*args, **kwargs):
        global TIME_DICT

        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True) 
        
        start = time.time()
        func_return = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()

        if func.__name__ in TIME_DICT:
            TIME_DICT[func.__name__]['time'] += round(end-start, 3)
            TIME_DICT[func.__name__]['count'] += 1
        else:
            TIME_DICT[func.__name__] = {'time': round(end-start, 3), 'count': 1}

        
        return func_return
    return deco



@elapse_time
def load_dataset(dataset_name, bs, img_size, rank=-1, world_size=1, data_aug=True, workers=8, data_dir='./data'):
    """
    get train/test dataloader
    """
    if rank in [-1, 0]: # for DistributedDataParallel (DDP)  
        logger.info(f'Loading {dataset_name} dataset ...')

    try: 
        dataset_func = getattr(dataset, dataset_name)
    except:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    train_dt, test_dt = dataset_func(data_aug, img_size, data_dir) # get train/test dataset 
    

    # for DDP
    tr_sampler = torch.utils.data.distributed.DistributedSampler(train_dt, num_replicas=world_size, rank=rank) if rank != -1 else None
    te_sampler = torch.utils.data.distributed.DistributedSampler(test_dt, num_replicas=world_size, rank=rank) if rank != -1 else None

    nw = min([os.cpu_count(), bs if bs > 1 else 0, workers])  # number of workers

    # get train/test loaders from train/test datasets 
    train_loader = DataLoader(train_dt, batch_size=bs // world_size, num_workers=nw, pin_memory=True, sampler=tr_sampler)
    test_loader = DataLoader(test_dt, batch_size=bs // world_size, num_workers=nw, pin_memory=True, sampler=te_sampler)
    
    return train_loader, test_loader

@elapse_time
def load_model(model_name, num_classes, device, rank=-1, pretrained_path=None):
    """
    get model you want to train
    """
    if rank in [-1, 0]: #for DDP
        logger.info(f'Loading {model_name} model ...\n')

    if pretrained_path == None:
        try:
            model_func = getattr(networks, model_name) 
        except:
            raise ValueError(f'Invalid model name: {model_name}')

        model = model_func(num_classes)
    else:
        model = torch.load(pretrained_path, map_location=device) # for pretrained model
    return model
    


def select_device(device='', batch_size=None, rank=-1):
    """
    select GPU device you try to use
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    s = ''
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        device_count = len(devices)  # device count
        if device_count > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % device_count == 0, f'batch-size {batch_size} not multiple of GPU count {device_count}'
        space = ' ' * (20)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    if rank in [-1, 0]:
        logger.info(s)
    return torch.device('cuda:0' if cuda else 'cpu')


def print_args(args):
    """
    print args
    """
    split_train_cfg = vars(args)
    
    num_dummy = 60
    train_txt = ' Train configuration '.center(num_dummy,' ')
    border_txt = '-'*num_dummy
    
    logger.info(f'+{border_txt}+')
    logger.info(f'|{train_txt}|')
    logger.info(f'+{border_txt}+')
    logger.info(f'|{" ".ljust(num_dummy)}|')
    for key, val in split_train_cfg.items():
        if isinstance(val, int) or isinstance(val, float):
            val = str(val)

        logger.info(f'| {key.center(int(num_dummy/2)-1)}:{val.center(int(num_dummy/2-1))}|')
    logger.info(f'|{" ".ljust(num_dummy)}|')
    logger.info(f'+{border_txt}+\n')




@contextmanager
def torch_distributed_zero_first(local_rank):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

