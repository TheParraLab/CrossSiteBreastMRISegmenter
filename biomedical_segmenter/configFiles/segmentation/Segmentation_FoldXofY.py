#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = '19093'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV/fineTuneKFold_*of*_testT1post.txt',
		   '/CV/fineTuneKFold_*of*_testSlope1.txt',
		   '/CV/fineTuneKFold_*of*_testSlope2.txt']

segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'UNet_Axl_FT_Nick' 
session = '' #NAME OF FINETUNE SESSION
segmentation_dpatch = [107,107,27,3]
model_patch_reduction = [38,38,18]
model_crop = 0 

use_coordinates = False

using_unet_breastMask = False
resolution = 'high' #changed from low

path_to_model = '' #PATH TO FINETUNED MODEL
session =  path_to_model.split('/')[-3]

percentile_normalization = False
########################################### TEST PARAMETERS
quick_segmentation = True
output_probability = True
OUTPUT_PATH = '' #PATH TO SAVE OUTPUT SEGMENTATIONS
save_as_nifti = True  
dice_compare = False
full_segmentation_patches = True
test_subjects = 7
n_fullSegmentations = test_subjects
list_subjects_fullSegmentation = range(0,test_subjects)
size_test_minibatches = 16
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

