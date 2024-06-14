#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
import numpy as np
import os
wd = os.getcwd()

#import tensorflow as tf
#config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list="0"
#tf.keras.backend.set_session(tf.Session(config=config))
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'SESSION6_19093'

############################## Load dataset #############################
 
TPM_channel = ''

trainChannels = ['/CV/fineTuneKFold_*of*_trainT1post.txt',
		 '/CV/fineTuneKFold_*of*_trainSlope1.txt',
		 '/CV/fineTuneKFold_*of*_trainSlope2.txt']

trainLabels   = '/CV/fineTuneKFold_*of*_trainlabels.txt'
    
testChannels  =  ['/CV/fineTuneKFold_*of*_testT1post.txt',
		 '/CV/fineTuneKFold_*of*_testSlope1.txt',
		 '/CV/fineTuneKFold_*of*_testSlope2.txt']

testLabels = ''
    

validationChannels = trainChannels
validationLabels = trainLabels
testChannels = validationChannels
testLabels = validationLabels
 
output_classes = 2
test_subjects = 7 #40
    
#-------------------------------------------------------------------------------------------------------------
 
# Parameters 

######################################### MODEL PARAMETERS
model = 'UNet_3D_axl_fineTune' # 'UNet_3D_axl' # 'UNet_3D_axl_recurrent_V2'
dpatch= [75,75,19]
segmentation_dpatch = [107, 107, 27, 3] # [91, 91, 27, 3] # CANNOT INCREASE THIS TO REDUCE MEMORY USE?????
model_patch_reduction = [38,38,18]
model_crop = 0 # 40 for normal model.

using_unet_breastMask = False
resolution = 'high'

L2 = 0
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice'

load_model = False

path_to_model = '' #PATH TO MODEL TO BE FINE TUNED

if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 1e-4 #5e-05 #5e-06  #1e-4 #1e-5  #1e-06
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 8 # 50 #8 #1
epochs = 15 #102 #52

#---- Dataset/Model related parameters ----
samplingMethod_train = 1 
samplingMethod_val = 1 
use_coordinates = False

merge_breastMask_model = False
path_to_breastMask_model = ''
Context_parameters_trainable = False

#Adjusts the proportion of sampled regions from benign being high intensity
sample_intensity_based = True
percentile_voxel_intensity_sample_benigns = 95

balanced_sample_subjects = False 		# SET TO FALSE WHEN TRAINING DATA HAS NO MALIGNANT/BENGING LABEL (breast mask model)
proportion_malignants_to_sample_train = .5
proportion_malignants_to_sample_val = 0.5
#------------------------------------------
n_subjects = 70 #4#2 #10 #5 #50 #48 #2 #40
n_patches = n_subjects*50 #480 #20 #320  #400*10
size_minibatches = 16

data_augmentation = False 
proportion_to_flip = 0.5
percentile_normalization = False #True
verbose = False 
quickmode = True # Train without validation. Full segmentation often but only report dice score (whole)
n_subjects_val = 70 #10 #5 #50 #48 #2 #40 
n_patches_val = 70*10 #n_subjects_val*50 #480 #20 # 320 400*10
size_minibatches_val = 16

########################################### TEST PARAMETERS
output_probability = True   # not thresholded network output for full scan segmentation
quick_segmentation = True
OUTPUT_PATH = ''
n_full_segmentations = 12
full_segmentation_patches = True
size_test_minibatches = 16
list_subjects_fullSegmentation = np.random.choice(range(0,35), size=n_full_segmentations, replace=False) # Leave empty if random
epochs_for_fullSegmentation = np.arange(1,epochs+1,1) #Defines the epochs at which to perform full segmentation and log results
saveSegmentation = False
propextracortion_malignants_fullSegmentation = 0.75

threshold_EARLY_STOP = 0 #Disabled when == 0

penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

