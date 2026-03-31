# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:14:47 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, GetTiles
from utils.core_utils import Train_model_Classic, Validate_model_Classic
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
import utils.utils as utils

from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import torch
import os
import random
from sklearn import preprocessing

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def Classic_Training(args):
    rank = args.rank
    world_size = args.world_size
        
    targetLabels = args.target_labels
    for targetLabel in targetLabels:
        for repeat in range(args.repeatExperiment): 
            args.target_label = targetLabel        
            random.seed(args.seed)
            args.projectFolder = utils.CreateProjectFolder(
                ExName = args.project_name, 
                ExAdr = args.adressExp, 
                targetLabel = targetLabel,
                model_name = args.model_name, 
                repeat = repeat + 1)

            if rank == 0:
                print('-' * 30 + '\n')
                print("Project Folder: ", args.projectFolder)
            
                if os.path.exists(args.projectFolder):
                    continue
                else:
                    os.mkdir(args.projectFolder) 
                
            args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
                
            if rank == 0:
                os.makedirs(args.result_dir, exist_ok = True)
                os.makedirs(args.split_dir, exist_ok = True)
             
            dist.barrier()

            reportFile = None
            if rank == 0:   
                reportFile = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
                reportFile.write('-' * 30 + '\n')
                reportFile.write(str(args))
                reportFile.write('-' * 30 + '\n')
            
                print('\nLOAD THE DATASET FOR TRAINING...\n')     


                patientsList, labelsList, csvFile = ConcatCohorts_Classic(
                    imagesPath = args.datadir_train, 
                    cliniTablePath = args.clini_dir, 
                    slideTablePath = args.slide_dir,
                    label = targetLabel, 
                    minNumberOfTiles = args.minNumBlocks,
                    outputPath = args.projectFolder, 
                    reportFile = reportFile, 
                    csvName = args.csv_name,
                    patientNumber = args.numPatientToUse)                    
            
                labelsList = utils.CheckForTargetType(labelsList)
                le = preprocessing.LabelEncoder()
                labelsList = le.fit_transform(labelsList)
            
                args.num_classes = len(set(labelsList))
                args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
                args.csvFile = csvFile
                utils.Summarize(args, list(labelsList), reportFile)
                

                if not args.train_full:
                    splits = list(StratifiedKFold(n_splits=args.k, random_state=args.seed, shuffle=True)
                                  .split(patientsList, labelsList))
                else:
                    splits = None

                if len(patientsList) < 20:
                    print('NOT ENOUGH DATA FOR TRAINING!')
                    continue

            else:
                patientsList, labelsList, splits, csvFile = None, None, None, None
                args.csvFile = None
                args.num_classes = None
                args.target_labelDict = None

            dist.barrier()
            shared_data = [patientsList, labelsList, splits, csvFile, args.num_classes, args.target_labelDict]
            dist.broadcast_object_list(shared_data, src=0)
            patientsList, labelsList, splits, csvFile, args.num_classes, args.target_labelDict = shared_data
            args.csvFile = csvFile

            if args.train_full:
                if rank == 0:
                    print('FULL TRAINING FOR ' + targetLabel)            
                    print('GENERATE NEW TILES...')                            
                train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, maxBlockNum = args.maxBlockNum, test = False)                
                train_x = list(train_data['TilePath'])
                train_y = list(train_data['yTrue'])  
                if args.early_stopping:
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    val_x = list(val_data['TilePath']) 
                    val_y = list(val_data['yTrue'])                  
                    train_data = train_data[~train_data['TilePath'].isin(val_x)]
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    if rank == 0:
                        val_data.to_csv(os.path.join(args.split_dir, 'ValSplit.csv'), index = False)
                else:
                    valGenerator = []
                
                if rank == 0:
                    train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit.csv'), index = False) 
                    print()
                    print('-' * 30)
                

                print("Initializing Model on GPU: ", rank)
                model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)
                model.to(rank) 
                model = DDP(model, device_ids = [rank], output_device = rank)

                params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0, #default 0 
                          'pin_memory' : False} #default False
            
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
                train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                trainGenerator = torch.utils.data.DataLoader(train_set, sampler=train_sampler, **params)

                if args.early_stopping:
                    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=True)
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    valGenerator = torch.utils.data.DataLoader(val_set, sampler=val_sampler, **params)
                            
                noOfLayers = 0
                for name, child in model.named_children():
                     noOfLayers += 1            
                cut = int (args.freeze_Ratio * noOfLayers)                
                ct = 0
                for name, child in model.named_children():
                    ct += 1
                    if ct < cut:
                        for name2, params in child.named_parameters():
                            params.requires_grad = False
            
                optimizer = utils.get_optim(model, args, params = False)            
                criterion = nn.CrossEntropyLoss()

                if rank == 0:
                    print('\nSTART TRAINING ...', end = ' ')
                    
                model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model,
                                                 trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                 criterion = criterion, optimizer = optimizer, args = args, fold = 'FULL')            
                
                if rank == 0:
                    print('-' * 30) 
                    torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel'))                
                    history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_acc_history, val_loss_history)), 
                                    columns =['train_loss', 'train_acc', 'val_acc', 'val_loss'])                
                    history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + '.csv'), index = False)
                    reportFile.close()
            
            else:
                if rank == 0:
                    print(str(args.k) + '-FOLD CROSS VALIDATION TRAINING FOR ' + targetLabel)

                for fold_idx, (train_index, test_index) in enumerate(splits):
                    trainPatients = np.array(patientsList)[train_index]
                    testPatients = np.array(patientsList)[test_index]

                    if rank == 0:
                        print(f"\nProcessing fold {fold_idx + 1} of {args.k}...")
                        print('GENERATE NEW TILES...\n')    
                        print('FOR TRAIN SET...\n')  
                    train_data = GetTiles(
                        csvFile=args.csvFile, label=targetLabel, target_labelDict=args.target_labelDict,
                        maxBlockNum=args.maxBlockNum, test=False, filterPatients=trainPatients)
                    dist.barrier()
                    if rank == 0:
                        print('FOR VALIDATION SET...\n') 
                    val_data = train_data.groupby('yTrue', group_keys=False).apply(lambda x: x.sample(frac=0.1))
                    train_data = train_data[~train_data['TilePath'].isin(val_data['TilePath'])]
                    dist.barrier()
                    if rank == 0:
                        print('FOR TEST SET...\n')
                    test_data = GetTiles(
                        csvFile=args.csvFile, label=targetLabel, target_labelDict=args.target_labelDict,
                        maxBlockNum=args.maxBlockNum, test=True, filterPatients=testPatients)
                    dist.barrier()
                    if rank == 0:
                        print('SAVE SPLITS...\n')
                        train_data.to_csv(os.path.join(args.split_dir, f'TrainSplit_{fold_idx + 1}.csv'), index=False)
                        val_data.to_csv(os.path.join(args.split_dir, f'ValSplit_{fold_idx + 1}.csv'), index=False)
                        test_data.to_csv(os.path.join(args.split_dir, f'TestSplit_{fold_idx + 1}.csv'), index=False)

                    dist.barrier()

                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue'])
                    val_x = list(val_data['TilePath'])
                    val_y = list(val_data['yTrue'])
                    test_x = list(test_data['TilePath'])
                    test_y = list(test_data['yTrue'])


    
                    
                    print("Initializing Model on GPU: ", rank)
                    model, input_size = utils.Initialize_model(args.model_name, args.num_classes, feature_extract = False, use_pretrained = True)
                    model.to(rank)
                    model = DDP(model, device_ids = [rank], output_device = rank)
                    params = {'batch_size': args.batch_size,
                              'shuffle': False,
                              'num_workers': 0,
                              'pin_memory' : False}
                
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
                    train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    trainGenerator = torch.utils.data.DataLoader(train_set, sampler=train_sampler, **params)
                
                    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    valGenerator = torch.utils.data.DataLoader(val_set, sampler=val_sampler, **params)   

                    params = {'batch_size': args.batch_size,
                              'shuffle': False,
                              'num_workers': 0, 
                              'pin_memory' : False}
                
                    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
                    test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                    testGenerator = torch.utils.data.DataLoader(test_set, sampler=test_sampler, **params)
                                      
                    noOfLayers = 0
                    for name, child in model.named_children():
                        noOfLayers += 1
                    
                    cut = int (args.freeze_Ratio * noOfLayers)

                    ct = 0
                    for name, child in model.named_children():
                        ct += 1
                        if ct < cut:
                            for name2, params in child.named_parameters():
                                params.requires_grad = False
                            
                    optimizer = utils.get_optim(model, args, params = False)                
                    criterion = nn.CrossEntropyLoss()
                
                    if rank == 0:
                        print('\n')
                        print('START TRAINING ...')
                    model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model, trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                     criterion = criterion, optimizer = optimizer, args = args, fold = str(fold_idx + 1), rank = rank)   
                             
                    print("Fold: ", fold_idx + 1)
                    print("Train_model_Classic finished on GPU: ", rank)
                    print("train_loss_history: ", train_loss_history)
                    print("train_acc_history: ", train_acc_history)
                    print("val_acc_history: ", val_acc_history)
                    print("val_loss_history: ", val_loss_history)
                    
                    dist.barrier()


                    # Result aggregation
                    max_length = max(len(train_loss_history), len(train_acc_history), len(val_acc_history), len(val_loss_history))
                    train_loss_history += [0] * (max_length - len(train_loss_history))
                    train_acc_history += [0] * (max_length - len(train_acc_history))
                    val_acc_history += [0] * (max_length - len(val_acc_history))
                    val_loss_history += [0] * (max_length - len(val_loss_history))

                    train_loss_history_all = [torch.zeros(max_length, dtype=torch.float32).to(rank) for _ in range(world_size)]
                    train_acc_history_all = [torch.zeros(max_length, dtype=torch.float32).to(rank) for _ in range(world_size)]
                    val_acc_history_all = [torch.zeros(max_length, dtype=torch.float32).to(rank) for _ in range(world_size)]
                    val_loss_history_all = [torch.zeros(max_length, dtype=torch.float32).to(rank) for _ in range(world_size)]

                    dist.all_gather(train_loss_history_all, torch.tensor(train_loss_history, dtype=torch.float32).to(rank))
                    dist.all_gather(train_acc_history_all, torch.tensor(train_acc_history, dtype=torch.float32).to(rank))
                    dist.all_gather(val_acc_history_all, torch.tensor(val_acc_history, dtype=torch.float32).to(rank))
                    dist.all_gather(val_loss_history_all, torch.tensor(val_loss_history, dtype=torch.float32).to(rank))

                    if rank == 0:
                        print('-' * 30)                
                        torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModelFold' + str(fold_idx + 1)))

                        combined_train_loss_history = torch.cat(train_loss_history_all, dim=0).tolist()
                        combined_train_acc_history = torch.cat(train_acc_history_all, dim=0).tolist()
                        combined_val_acc_history = torch.cat(val_acc_history_all, dim=0).tolist()
                        combined_val_loss_history = torch.cat(val_loss_history_all, dim=0).tolist()

                        history = pd.DataFrame(list(zip(combined_train_loss_history, combined_train_acc_history, combined_val_loss_history, combined_val_acc_history)), 
                                    columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])

                        #history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                        #            columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
                    
                        history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FOLD_' + str(fold_idx + 1) + '.csv'), index = False)
                        print('\nSTART EVALUATION ON TEST DATA SET ...', end = ' ')
                        
                        model.load_state_dict(torch.load(os.path.join(args.projectFolder, 'RESULTS', 'bestModelFold' + str(fold_idx + 1))))
                        probsList  = Validate_model_Classic(model = model, dataloaders = testGenerator)

                        probs = {}
                        for key in list(args.target_labelDict.keys()):
                            probs[key] = []
                            for item in probsList:
                                probs[key].append(item[utils.get_value_from_key(args.target_labelDict, key)])
                    
                        probs = pd.DataFrame.from_dict(probs)
                        testResults = pd.concat([test_data, probs], axis = 1)                    
                        testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FOLD_' + str(fold_idx + 1) + '.csv')
                        testResults.to_csv(testResultsPath, index = False)
                        CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = fold_idx + 1 , clamMil = False, reportFile = reportFile)                         
                        reportFile.write('-' * 30 + '\n')                               

                if rank == 0:
                    patientScoreFiles = []
                    tileScoreFiles = []                
                    for i in range(args.k):
                        patientScoreFiles.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i+1) + '.csv')
                        tileScoreFiles.append('TEST_RESULT_TILE_BASED_FOLD_' + str(i+1) + '.csv')      
                    CalculateTotalROC(resultsPath = args.result_dir, results = patientScoreFiles, target_labelDict =  args.target_labelDict, reportFile = reportFile) 
                    reportFile.write('-' * 30 + '\n')
                    MergeResultCSV(args.result_dir, tileScoreFiles)
                    reportFile.close()
    
    dist.barrier() # Sync all processes
##############################################################################





















