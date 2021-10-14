import os
import copy
from abc import ABC, abstractmethod
from ptflops import get_model_complexity_info
from thop import profile 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from log import get_logger

logger = get_logger(__name__)

class TrainerBase(ABC):
    def __init__(
        self,
        args,
        model,
        train_loader,
        test_loader,
        device):

        self.device_count = len(args.device.split(','))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.img_size = args.img_size
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.epochs = args.epochs

        file_name = f'{self.dataset_name}_{self.model_name}.pt'
        self.save_path = os.path.join(args.save_dir, file_name)

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def _get_file_size(self, file_path):
        size = os.path.getsize(file_path)
        return size

    def model_summary(self, test_acc, test_loss, model):
        model = copy.deepcopy(model)

        input_image = torch.randn(1, 3, self.img_size, self.img_size).cuda()

        if self.device_count > 1 and hasattr(model, 'module'):
            flops, params = profile(model.module.cuda(), inputs=(input_image,), verbose=False)
        else:
            flops, params = profile(model, inputs=(input_image,), verbose=False)
        
        del model, input_image
        model_memory = self._get_file_size(self.save_path)

        acc_txt =           ' Test Accuracy (%)            '
        loss_txt =          ' Test loss                    '
        param_num_txt =     ' Number of parameters (M)     '
        flop_txt =          ' Flops                        '
        file_size_txt =     ' File size (MB)               '

        len_txt = len(acc_txt)
        
        total_dummy = 50
        l_dummy = len_txt
        r_dummy = total_dummy - len_txt - 1
        edge_line = '-' * total_dummy

        model_name = ' '+self.model_name+'_model  '
        
        acc =           f' {round(test_acc*100.0, 2)} %'
        loss =          f' {round(test_loss, 4)}'
        param_num =     f' {round(params * 1e-6, 2)} M'
        flops =         f' {round(flops * 1e-6, 2)} M'
        file_size =     f' {round(model_memory / (1024 * 1024), 2)} MB'

        logger.info(f'+{edge_line}+')
        logger.info(f'|{" ".ljust(l_dummy)}|{model_name.ljust(r_dummy)}|')
        logger.info(f'+{edge_line}+')
        logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|')
        logger.info(f'|{acc_txt}|{acc.ljust(r_dummy)}|')
        logger.info(f'|{loss_txt}|{loss.ljust(r_dummy)}|')
        logger.info(f'|{param_num_txt}|{param_num.ljust(r_dummy)}|')
        logger.info(f'|{flop_txt}|{flops.ljust(r_dummy)}|')
        logger.info(f'|{file_size_txt}|{file_size.ljust(r_dummy)}|')
        logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|')
        logger.info(f'+{edge_line}+\n')
