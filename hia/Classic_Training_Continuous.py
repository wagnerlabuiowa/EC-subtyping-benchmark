import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Classic_Training_Regression(args):
    targetLabels = args.target_labels
    for targetLabel in targetLabels:
        for repeat in range(args.repeatExperiment): 
            
            args.target_label = targetLabel        
            random.seed(args.seed)
            args.projectFolder = utils.CreateProjectFolder(ExName = args.project_name, ExAdr = args.adressExp, targetLabel = targetLabel,
                                                           model_name = args.model_name, repeat = repeat + 1)
            print('-' * 30 + '\n')
            print(args.projectFolder)
            if os.path.exists(args.projectFolder):
                continue
            else:
                os.mkdir(args.projectFolder) 
                
            args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
            os.makedirs(args.result_dir, exist_ok = True)
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
               
            reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
            reportFile.write('-' * 30 + '\n')
            reportFile.write(str(args))
            reportFile.write('-' * 30 + '\n')
            
            print('\nLOAD THE DATASET FOR TRAINING...\n')     
            patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_train, 
                                                                          cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                          label = targetLabel, minNumberOfTiles = args.minNumBlocks,
                                                                          outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name,
                                                                          patientNumber = args.numPatientToUse)                        
            labelsList = utils.CheckForTargetType(labelsList)
            
            # No label encoding needed for continuous variables
            labelsList = np.array(labelsList).astype(np.float32)
            
            args.num_classes = 1  # For regression, we predict a single continuous value
            
            utils.Summarize(args, list(labelsList), reportFile)
        
            if len(patientsList) < 20:
                print('NOT ENOUGH DATA FOR TRAINING!')
                continue

            if args.train_full:
                print('IT IS A FULL TRAINING FOR ' + targetLabel + '!')            
                print('GENERATE NEW TILES...')                            
                train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = None, maxBlockNum = args.maxBlockNum, test = False)                
                train_x = list(train_data['TilePath'])
                train_y = list(train_data['yTrue'])  

                if args.early_stopping:
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    val_x = list(val_data['TilePath']) 
                    val_y = list(val_data['yTrue'])                  
                    train_data = train_data[~train_data['TilePath'].isin(val_x)]
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    val_data.to_csv(os.path.join(args.split_dir, 'ValSplit.csv'), index = False)
                else:
                    valGenerator = []
                    
                train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit.csv'), index = False)                                              
                print()
                print('-' * 30)
                
                model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True, regression=True)
                model.to(device) 
                
                params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0,
                          'pin_memory' : False}
            
                train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size, regression=True)           
                trainGenerator = torch.utils.data.DataLoader(train_set, **params)
                if args.early_stopping:
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size, regression=True)           
                    valGenerator = torch.utils.data.DataLoader(val_set, **params)
                            
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
                criterion = nn.MSELoss()
                print('\nSTART TRAINING ...', end = ' ')
                model, train_loss_history, val_loss_history = Train_model_Classic(model = model,
                                                 trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                 criterion = criterion, optimizer = optimizer, args = args, fold = 'FULL', regression=True)            
                print('-' * 30)
                        
                torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel'))                
                history = pd.DataFrame(list(zip(train_loss_history, val_loss_history)), 
                                  columns =['train_loss', 'val_loss'])                
                history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + '.csv'), index = False)
                reportFile.close()
            
            else:
                
                print('IT IS A ' + str(args.k) + 'FOLD CROSS VALIDATION TRAINING FOR ' + targetLabel + '!')
                patientID = np.array(patientsList)
                labels = np.array(labelsList)
            
                folds = args.k
                kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
                kf.get_n_splits(patientID, labels)
                
                foldcounter = 1
            
                for train_index, test_index in kf.split(patientID, labels):
                  
                    testPatients = patientID[test_index]   
                    trainPatients = patientID[train_index] 
                    
                    print('GENERATE NEW TILES...\n')    
                    print('FOR TRAIN SET...\n')                         
                    train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = None, maxBlockNum = args.maxBlockNum, test = False, filterPatients = trainPatients)                
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    print('FOR VALIDATION SET...\n')  
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    val_x = list(val_data['TilePath']) 
                    val_y = list(val_data['yTrue'])                  
                    train_data = train_data[~train_data['TilePath'].isin(val_x)]
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    print('FOR TEST SET...\n')
                    test_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = None, maxBlockNum = args.maxBlockNum, test = True, filterPatients = testPatients)                                    
                    test_x = list(test_data['TilePath'])
                    test_y = list(test_data['yTrue']) 
                    
                    test_data.to_csv(os.path.join(args.split_dir, 'TestSplit_' + str(foldcounter) + '.csv'), index = False)
                    train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit_' + str(foldcounter) + '.csv'), index = False)
                    val_data.to_csv(os.path.join(args.split_dir, 'ValSplit_' + str(foldcounter) + '.csv'), index = False)                       

                    print('-' * 30)
                    print("K FOLD VALIDATION STEP => {}".format(foldcounter))  
                    print('-' * 30)  
                    
                    model, input_size = utils.Initialize_model(args.model_name, args.num_classes, feature_extract = False, use_pretrained = True, regression=True)
                    model.to(device)
                    params = {'batch_size': args.batch_size,
                              'shuffle': True,
                              'num_workers': 0,
                              'pin_memory' : False}
                
                    train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size, regression=True)           
                    trainGenerator = torch.utils.data.DataLoader(train_set, **params)
                
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size, regression=True)           
                    valGenerator = torch.utils.data.DataLoader(val_set, **params)   

                    params = {'batch_size': args.batch_size,
                              'shuffle': False,
                              'num_workers': 0, 
                              'pin_memory' : False}
                
                    test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size, regression=True)
                    testGenerator = torch.utils.data.DataLoader(test_set, **params)
                                      
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
                    criterion = nn.MSELoss()
                
                    print('\n')
                    print('START TRAINING ...')
                    model, train_loss_history, val_loss_history = Train_model_Classic(model = model, trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                     criterion = criterion, optimizer = optimizer, args = args, fold = str(foldcounter), regression=True)            
                    print('-' * 30)
                
                    torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModelFold' + str(foldcounter)))
                    history = pd.DataFrame(list(zip(train_loss_history, val_loss_history)), 
                                  columns =['train_loss', 'val_loss'])
                
                    history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FOLD_' + str(foldcounter) + '.csv'), index = False)
                    print('\nSTART EVALUATION ON TEST DATA SET ...', end = ' ')
                    
                    model.load_state_dict(torch.load(os.path.join(args.projectFolder, 'RESULTS', 'bestModelFold' + str(foldcounter))))
                    predsList = Validate_model_Classic(model = model, dataloaders = testGenerator, regression=True)

                    test_data['Preds'] = predsList
                    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FOLD_' + str(foldcounter) + '.csv')
                    test_data.to_csv(testResultsPath, index = False)
                    
                    mse = mean_squared_error(test_data['yTrue'], test_data['Preds'])
                    reportFile.write(f'Fold {foldcounter} MSE: {mse}\n')
                    reportFile.write('-' * 30 + '\n')                
                    foldcounter +=  1               
            
                reportFile.close()

def Validate_model_Classic(model, dataloaders, regression=False):
    model.eval()
    predsList = []

    with torch.no_grad():
        for inputs, _ in dataloaders:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if regression:
                preds = outputs.squeeze().cpu().numpy()
            else:
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
            predsList.extend(preds)
    
    return predsList
