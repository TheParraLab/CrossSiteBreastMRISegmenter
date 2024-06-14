"""
get_paths.py

This script contains functions for manipulating and validating lists of data, 
specifically for the purpose of preparing data for k-fold cross-validation.

Running this script will generate text files containing paths to data for each fold of cross-validation.
These files are used by the biomedical_segmenter package to train and test models.
Parameters:
- SAVE: If True, paths will be saved to text files.
- kfold: If True, data will be split into k folds for cross-validation.
- k: Number of folds for cross-validation.
- FILE_LABEL: Root name for output files.
- FILE_PATH: Project directory for output paths.
- DATA_PATH: Path to data directory.
- MASK_PATH: Path to ground truth (mask) directory.
- seg_suffix: Suffix for ground truth (mask) files.
- blacklist: List of IDs to exclude from data.

Functions:
- prune_list: Removes items in a blacklist from the main list.
- prepare_kfold: Splits data into k folds for cross-validation.
- validate_unique_groups: Ensures no ID is present in more than one group.

The script also includes code to set the working directory, specify parameters for k-fold cross-validation, 
and identify and prune specific IDs from the data directory.

Note: This script assumes it is run from a directory within the 'Axial_Crosssite' project.

@author: Nicholas Leotta
@last updated: 12/19/2023
"""

import os
import random

def prune_list(main, blacklist, display=True):
    #Remove items in blacklist from main list
    for item in blacklist:
        if item in main:
            if display:
                print('Removing '+item+' from list')
            main.remove(item)
    return main

def prepare_kfold(data, k):
    #Split data into k folds
    data_split = []
    for i in range(k):
        data_split.append([])
    for i in range(len(data)):
        data_split[i%k].append(data[i])
    return data_split

def validate_unique_groups(data_split):
    #Ensure no ID is present in more than one group
    all_items = []
    for group in data_split:
        for item in group:
            if item in all_items:
                raise Exception('Duplicate item found in groups')
            all_items.append(item)
    print('All groups unique')

#Force code to execute with the main project directory as working directory
wdir = os.getcwd()
while wdir.split('/')[-1].lower() != 'axial_crosssite_release':
    os.chdir('..')
    wdir = os.getcwd()
print(f'Working directory: {wdir}')

#Parameters
SAVE = True #Save paths to text files
kfold = False #Perform k-fold cross-validation
k = 6 #Number of folds
FILE_LABEL = 'PublicData' #Root name for output files
FILE_PATH = './biomedical_segmenter/CV/' #Project directory for output paths
DATA_PATH = './biomedical_segmenter/Data/PublicData/Data' #Path to data directory
MASK_PATH = './biomedical_segmenter/Data/PublicData/Segmentation' #Path to ground truth (mask) directory 
seg_suffix = '_T1subtract.nii.gz' #Suffix for ground truth (mask) files
blacklist = [] #List of IDs to exclude from data

#Ensure FILE_PATH exists
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)
if DATA_PATH[-1] != '/':
    DATA_PATH += '/'
if MASK_PATH[-1] != '/':
    MASK_PATH += '/'
if DATA_PATH[0] == '.':
    DATA_PATH = wdir+DATA_PATH[1:]
if MASK_PATH[0] == '.':
    MASK_PATH = wdir+MASK_PATH[1:]
    
#Identifying IDs from data directory
IDs = os.listdir(DATA_PATH)
print(f'Found {len(IDs)} IDs')

#Applying blacklist
IDs = prune_list(IDs, blacklist)
print(f'Pruned {len(blacklist)} IDs from list')

#Appending data directory to IDs
paths = [DATA_PATH+ID for ID in IDs]
print(f'Formatted paths for {len(paths)} IDs')
random.shuffle(paths)
if kfold:
    IDs_split = prepare_kfold(paths, 6)
    print(f'IDs split into {len(IDs_split)} groups of {len(IDs_split[0])} IDs')
    validate_unique_groups(IDs_split)

channels = [ 'Slope1', 'Slope2', 'T1post']
#files dictionary converts from channel name to respective file name for each channel
files = {'slope1': 'T1_axial_slope1.nii',
         'slope2': 'T1_axial_slope2.nii',
         't1post': 'T1_axial_02.nii'}

#Writing paths to text files
if SAVE:
    path_orig = paths.copy()

    if kfold:   #Writing paths for k-fold cross-validation
        for i in range(len(IDs_split)):   #For each fold
            paths = path_orig.copy()
            print(f'Writing paths for fold {i+1} of {len(IDs_split)}')
            for channel in channels:   #Writing paths for each channel

                #Writing training paths
                with open(FILE_PATH+FILE_LABEL+f'_{i+1}of{len(IDs_split)}_train{channel}.txt', 'w') as f:
                    for path in prune_list(paths, IDs_split[i], False):
                        f.write(path+'/'+files[channel.lower()]+'\n')

                #Writing testing paths
                with open(FILE_PATH+FILE_LABEL+f'_{i+1}of{len(IDs_split)}_test{channel}.txt', 'w') as f:
                    for path in IDs_split[i]:
                        f.write(path+'/'+files[channel.lower()]+'\n')
            
            #Writing training mask paths
            paths = path_orig.copy()
            with open(FILE_PATH+FILE_LABEL+f'_{i+1}of{len(IDs_split)}_trainlabels.txt', 'w') as f:
                for path in prune_list(paths, IDs_split[i], False):
                    f.write(MASK_PATH+path.split('/')[-1]+seg_suffix+'\n')

    else:   #Writing paths for all data
        for channel in channels:

            #Writing training paths
            with open(FILE_PATH+FILE_LABEL+f'_all{channel}.txt', 'w') as f:
                for path in paths:
                    f.write(path+'/'+files[channel.lower()]+'\n')
        
        #Writing training mask paths
        with open(FILE_PATH+FILE_LABEL+f'_alltrainlabels.txt', 'w') as f:
            for path in paths:
                f.write(MASK_PATH+path.split('/')[-1]+seg_suffix+'\n')
