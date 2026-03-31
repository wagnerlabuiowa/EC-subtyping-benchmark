# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:45:05 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

import utils.utils as utils
from utils.core_utils import Validate_model_Classic
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, GetTiles
from eval.eval import CalculatePatientWiseAUC, GenerateHighScoreTiles_Classic
import torch.nn as nn
import torchvision
import pandas as pd
import argparse
import torch
import os
import random
import collections
from sklearn import preprocessing
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

##############################################################################

def setup_ddp():
    """Initializes the DDP process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the DDP process group."""
    dist.destroy_process_group()

def run_deployment(rank, world_size, args):
    """The main deployment logic for a single DDP process."""
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f'\nTORCH Detected: {device} with {world_size} GPUs\n')
        # Enable half-precision inference
        torch.backends.cudnn.benchmark = True  # Speed up fixed-size inputs

    ##############################################################################
    
    args = utils.ReadExperimentFile(args, deploy = True)    
    random.seed(args.seed)        
    args.target_label = args.target_labels[0]  

    # All ranks determine the project folder paths
    args.projectFolder = utils.CreateProjectFolder(ExName=args.project_name, ExAdr=args.adressExp, targetLabel=args.target_label,
                                                   model_name=args.model_name)
    args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
    args.split_dir = os.path.join(args.projectFolder, 'SPLITS')

    reportFile = None
    if rank == 0:
        print('-' * 30 + '\n')
        print(args.projectFolder)
        if os.path.exists(args.projectFolder):
            print('THIS FOLDER ALREADY EXITS!!! PLEASE REMOVE THE FOLDER, IF YOU WANT TO RE-RUN.')
        else:
            os.makedirs(args.projectFolder, exist_ok=True)

        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.split_dir, exist_ok=True)

        reportFile = open(os.path.join(args.projectFolder, 'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('-' * 30 + '\n')
        reportFile.write(str(args))
        reportFile.write('-' * 30 + '\n')

    if rank == 0:
        print('\nLOAD THE DATASET FOR TESTING...\n')

    if rank == 0:
        # Rank 0 performs all data preparation and file I/O
        patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath=args.datadir_test,
                                                                       cliniTablePath=args.clini_dir, slideTablePath=args.slide_dir,
                                                                       label=args.target_label, minNumberOfTiles=args.minNumBlocks,
                                                                       outputPath=args.projectFolder, reportFile=reportFile, csvName=args.csv_name,
                                                                       patientNumber=args.numPatientToUse)
        labelsList = utils.CheckForTargetType(labelsList)
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))

        utils.Summarize(args, list(labelsList), reportFile)
        print('-' * 30)
        print('IT IS A DEPLOYMENT FOR ' + args.target_label + '!')
        print('GENERATE NEW TILES...')

        test_data = GetTiles(csvFile=args.csvFile, label=args.target_label, target_labelDict=args.target_labelDict,
                             maxBlockNum=args.maxBlockNum, test=True)
        test_x = list(test_data['TilePath'])
        test_y = list(test_data['yTrue'])

        test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index=False)
        print()
        print('-' * 30)
        
        # Package the necessary data for other processes
        data_to_broadcast = [test_x, test_y, args.num_classes, args.target_labelDict, test_data]
    else:
        # Other processes will receive this data
        data_to_broadcast = [None, None, None, None, None]

    # Broadcast the data from rank 0 to all other processes
    dist.broadcast_object_list(data_to_broadcast, src=0)
    
    # Unpack the data on all processes
    test_x, test_y, args.num_classes, args.target_labelDict, test_data = data_to_broadcast
            
    model, input_size = utils.Initialize_model(model_name=args.model_name, num_classes=args.num_classes,
                                               feature_extract=False, use_pretrained=True)
    if rank == 0:
        print("Model Address before loading: ", args.modelAdr)

    # Load model on CPU first to avoid GPU memory spike on rank 0
    state_dict = torch.load(args.modelAdr, map_location='cpu')
    if isinstance(state_dict, collections.OrderedDict):
        model.load_state_dict(state_dict)
    else:
        model = state_dict
        
    model.to(device)
    model = DDP(model, device_ids=[rank])
    
    if device.type == 'cuda':
        model.module.half()
        transform_fn = lambda x: torchvision.transforms.functional.to_tensor(x).half()
    else:
        transform_fn = torchvision.transforms.ToTensor()

    if rank == 0:
        print("test set datasetloader_classic")  
        
    test_set = DatasetLoader_Classic(test_x, test_y, transform=transform_fn, target_patch_size=input_size)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    params = {'batch_size': args.batch_size,
              'num_workers': args.num_workers, # Make sure num_workers is set in your args
              'pin_memory' : True,
              'sampler': test_sampler}

    if rank == 0:
        print("test set generator")    
    testGenerator = torch.utils.data.DataLoader(test_set, **params)
    
    if rank == 0:
        print('START DEPLOYING...')
        print('')
    
    # Each process gets its own list of probabilities
    local_probsList  = Validate_model_Classic(model = model, dataloaders = testGenerator)

    # Save local results to disk
    local_probs_df = pd.DataFrame(local_probsList)
    local_probs_path = os.path.join(args.result_dir, f"probs_rank{rank}.csv")
    local_probs_df.to_csv(local_probs_path, index=False)

    # Synchronize all ranks
    dist.barrier()

    if rank == 0:
        # Merge all per-rank results
        all_probs = []
        for r in range(world_size):
            path = os.path.join(args.result_dir, f"probs_rank{r}.csv")
            df = pd.read_csv(path)
            all_probs.append(df)
        probsList = pd.concat(all_probs, ignore_index=True)

        # Now build the probs dict as before
        probs = {}
        for key in list(args.target_labelDict.keys()):
            probs[key] = []
            for item in probsList.values:
                # item is a row, get the correct column index for the key
                col_idx = list(args.target_labelDict.keys()).index(key)
                probs[key].append(item[col_idx])
        probs = pd.DataFrame.from_dict(probs)

        # Handle potential padding from DistributedSampler
        actual_dataset_size = len(test_data)
        if len(probs) > actual_dataset_size:
            print(f"Warning: DistributedSampler added {len(probs) - actual_dataset_size} padding samples. Trimming results.")
            probs = probs.iloc[:actual_dataset_size]
            test_data_aligned = test_data
        else:
            test_data_aligned = test_data

        testResults = pd.concat([test_data_aligned.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
        
        testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FULL.csv')
        testResults.to_csv(testResultsPath, index = False)
        
        print(f"Saved results for {len(testResults)} samples to {testResultsPath}")
        
        totalPatientResultPath = CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = None ,
                                                        clamMil = False, reportFile = reportFile)  
        GenerateHighScoreTiles_Classic(totalPatientResultPath = totalPatientResultPath, totalResultPath = testResultsPath, 
                            numHighScorePetients = args.numHighScorePatients, numHighScoreTiles = args.numHighScorePatients,
                            target_labelDict = args.target_labelDict, savePath = args.result_dir)       
        reportFile.write('-' * 100 + '\n')
        print('\n')
        print('-' * 30)
        reportFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
    parser.add_argument('--adressExp', type = str, required=True, help = 'Address to the experiment File')
    parser.add_argument('--modelAdr', type = str, required=True, help = 'Address to the selected model')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    run_deployment(rank, world_size, args)
    
    cleanup_ddp() 