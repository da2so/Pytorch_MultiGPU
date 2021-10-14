import os
import time
import argparse
from tabulate import tabulate

from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from log import get_logger, setup_logger
from trainer_base import TrainerBase
from common import load_dataset, load_model, select_device, print_args, elapse_time, get_time_list

logger = get_logger(__name__)



class Classification_DP(TrainerBase):
    #@ElapseTime
    def __init__(self, args, model, train_loader, test_loader, device):
        super().__init__(args, model, train_loader, test_loader, device)

        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)


    @elapse_time
    def train(self, epoch):
        
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        total_correct = 0
        batches = 0
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader)-1, leave=False)
        for i, (images, labels) in loop:
            batches += len(labels)

            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            #print(f'Output: {outputs}')
            #print(f'Output size: {outputs.size()}')
            loss = self.loss(outputs, labels)

            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            loss.backward()

            self.optimizer.step()

            acc = float(total_correct) / batches
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Accuracy=acc, Loss=loss.item())

            if i == len(self.train_loader) -1:
                logger.debug(f'Train - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {loss.item()}')

        self.scheduler.step()
    
    @elapse_time
    def test(self, epoch, print_log=True):

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                avg_loss += self.loss(outputs, labels)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= (i+1)
        acc = float(total_correct) / len(self.test_loader.dataset)
        
        if epoch != -1 and print_log == True:
            logger.debug(f'Test  - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {avg_loss.data.item()}')

        return acc, avg_loss.data.item()
    
    def evaluate(self, print_log=True):
        return self.test(-1, print_log)

    @elapse_time
    def build(self):

        for epoch in range(1, self.epochs+1):
            self.train(epoch)
            test_acc, test_loss = self.test(epoch)

        logger.info(f'The trained model is saved in {self.save_path}\n')        
        torch.save(self.model, self.save_path)
        
        summary_dict = self.model_summary(test_acc, test_loss, self.model)
        


def main(args):

    args.save_dir = setup_logger('DP')

    print_args(args)
    device = select_device(args.device, args.bs)
    #Load dataset
    train_loader, test_loader = load_dataset(dataset_name=args.dataset,
                                            bs=args.bs,
                                            img_size=args.img_size)
    #Load model
    model = load_model(model_name=args.model,
                        num_classes=args.num_classes,
                        device=device)
    model = nn.DataParallel(model) #DP

    #Train and Test
    trainer = Classification_DP(args=args,
                                model=model,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                device=device)
    trainer.build()

    logger.info(f'\n{tabulate(get_time_list(), headers=["Fucntion", "Time Elapse(s)"], tablefmt="fancy_grid")}')

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch SingleGPU')

    parser.add_argument('--model', type = str, default = "resnet20", help = 'model name')
    parser.add_argument('--dataset', type = str, default = "cifar10", help = 'dataset name')
    parser.add_argument('--num_classes', type = int, default = 10, help = 'number of classes')
    parser.add_argument('--img_size', type = int, default = 32, help = 'image shape')

    parser.add_argument('--device', type = str, default = "3,4,5,6", help = 'cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--epochs', type = int, default = 10, help = 'epochs')

    parser.add_argument('--bs', type = int, default = 256, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 0.1, help = 'learning rate')

    args = parser.parse_args()
   
    main(args)