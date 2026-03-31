# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:08:57 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

from Classic_Training_DDP import Classic_Training

import utils.utils as utils

import warnings
import argparse
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"D:/NAME.txt", help = 'Address to the experiment File')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
############################################### ################################

def setup_ddp(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master node's address
    os.environ['MASTER_PORT'] = '12355'     # Set an open port number
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)

    args = utils.ReadExperimentFile(args)
    
    args.rank = rank
    args.world_size = world_size

    Classic_Training(args)
        
    cleanup_ddp()


if __name__ == '__main__':
    # Initialize distributed training
    gpu_count = torch.cuda.device_count()
    print('\nTORCH Detected: {} with {} GPU(s)\n'.format(device, gpu_count))
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)   
         
        
        
        
        
        
        
        
        
        
        
        
        