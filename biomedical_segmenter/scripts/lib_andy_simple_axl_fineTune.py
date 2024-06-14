#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Apr 21, 2023

@author: andy adapted from lukas - Upgraded to python3 by
"""

import os
from shutil import copy
import sys
import nibabel as nib
import numpy as np
np.set_printoptions(precision=3)
import time
import random
import gc

#import keras 
import pandas as pd
from skimage.transform import resize
from numpy.random import seed
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import pdb
from keras.optimizers import Adam

seed(1)
tf.random.set_seed(2)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

import multiprocessing        
from multiprocessing import Pool, Process


import ctypes
from ctypes import c_char_p
from ctypes import c_float
from ctypes import c_int

class scalingLayer(tf.keras.layers.Layer):

  def __init__(self):
      super(scalingLayer, self).__init__()

  def build(self, input_shape):  # Create the state of the layer (weights)
      self.w = tf.Variable(initial_value=[0.0,1.0,0.0],trainable=True,name='weights')
      self.b = tf.Variable(initial_value=[0.0,0.0,0.0],trainable=True,name='biases')

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return self.w * inputs + self.b

def myReadNiftiPatch(img,D1,D2,D3,dpatch):
    if len(dpatch)>3:
        dpatch = dpatch[0:3]

    low = (c_int*3)(D1-(dpatch[0]/2),D2-(dpatch[1]/2),D3-(dpatch[2]/2))
    high = (c_int*3)(D1+(dpatch[0]/2),D2+(dpatch[1]/2),D3+(dpatch[2]/2))
#    NOTE: AS THE C CODE IS FOLLOWING MATLAB STYLE, IT INCLUDES THE UP LIMIT IN INDEXING, SO HERE REMOVE THE ADDITIVE TERM IN "high", WHICH IS PYTHON STYLE (NOT INCLUDE UP LIMIT IN INDEXING)
    numOfVoxels = dpatch[0]*dpatch[1]*dpatch[2]
    fun = ctypes.CDLL('/home/andy/projects/lukasSegmenter/biomedical_segmenter/scripts/my_nifti1_read_patch.so')
    fun.main.restype = ctypes.POINTER(ctypes.c_float)
#    print('loading from voxel {} in subj {}'.format([D1,D2,D3],img))
#    x = fun.main(c_char_p(img),byref(low),byref(high),numOfVoxels)
    x = fun.main(c_char_p(img),low,high,numOfVoxels)
#    print('failed at {} in subj {}'.format([D1,D2,D3],img))
    xx = np.ctypeslib.as_array(x,shape=(numOfVoxels,1))
#    print('read done')
    return xx.reshape(dpatch,order='F')

################################## AUXILIARY FUNCTIONS #####################################

  
def flip_random(patches, labels, TPM_patches, coords, proportion_to_flip=0.5):

  # Why only flipping malignants???
  #malignant_indexes = np.argwhere(np.sum(np.sum(labels, axis=-1), axis=-1) > 0)[:,0]
  #indx_toflip = np.random.choice(malignant_indexes, int(len(malignant_indexes)*proportion_to_flip), replace=False)
  
  indx_toflip = np.random.choice(range(patches.shape[0]), int(patches.shape[0]*proportion_to_flip), replace=False)
  axis = np.random.choice(range(0,3),size=len(indx_toflip))
  for i in range(len(indx_toflip)):
    if axis[i] == 0:
      # SAGITTAL FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],::-1,:,:,ch]
      for ch in range(labels.shape[-1]):
          labels[indx_toflip[i],:,:,:,ch] = labels[indx_toflip[i],::-1,:,:,ch]
    elif axis[i] == 1:
      # CORONAL FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],:,::-1,:,ch]
      for ch in range(labels.shape[-1]):
          labels[indx_toflip[i],:,:,:,ch] = labels[indx_toflip[i],:,::-1,:,ch]
      if len(TPM_patches) != 0:
          TPM_patches[indx_toflip[i],0,:] = np.flip(TPM_patches[indx_toflip[i],0,:],0)
      if len(coords) != 0:
          for ch in range(coords.shape[-1]):
              coords[indx_toflip[i],0,:,:,ch] = np.flip(coords[indx_toflip[i],0,:,:,ch],0)
    elif axis[i] == 2:
      # AXIAL FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],:,:,::-1,ch]
      for ch in range(labels.shape[-1]):
          labels[indx_toflip[i],:,:,:,ch] = labels[indx_toflip[i],:,:,::-1,ch]
      if len(TPM_patches) != 0:
          TPM_patches[indx_toflip[i],0,:] = np.flip(TPM_patches[indx_toflip[i],0,:],1)        
      if len(coords) != 0:
          for ch in range(coords.shape[-1]):
              coords[indx_toflip[i],0,:,:,ch] = np.flip(coords[indx_toflip[i],0,:,:,ch],1)

  return patches, labels , TPM_patches
  
# ################################## SAMPLING FUNCTIONS #####################################


def generateRandomIndexesSubjects(n_subjects, total_subjects):
    indexSubjects = random.sample(range(total_subjects), n_subjects)
    return indexSubjects

def getSubjectChannels(subjectIndexes, channel):
    "With the channels (any modality) and the indexes of the selected subjects, return the addresses of the subjects channels"
    fp = open(channel)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i][:-1] for i in subjectIndexes]
    fp.close()
    return selectedSubjects


def getSubjectShapes_parallelization(subjectChannel,resolution):

#    subjectChannel = args[0]
#    resolution = args[1]
    proxy_img = nib.load(subjectChannel)
    res = proxy_img.header['pixdim'][1:4]
    shape = proxy_img.shape
    
    if resolution == 'high':
        if res[0] > 1.0:    
          target_res = [res[0]/2.,res[1]/2.,res[2]]
          out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
        else:
          out_shape = shape
    elif resolution == 'low':
        if res[0] < 0.5:    
          target_res = [res[0]*2.,res[1]*2.,res[2]]
          out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
        else:
          out_shape = shape
          
#    out_shape = shape        
    return out_shape    

def generateVoxelIndexes_parallel(subjectIndexes,CV_FOLDS_ARRAYS_PATH, target_shape, patches_per_subject, dpatch, n_patches, channels, channel_mri, samplingMethod, output_classes, percentile_voxel_intensity_sample_benigns ,percentile_normalization , allForegroundVoxels = "", verbose=False):
    allVoxelIndexes = {} #{a:None for a in subjectIndexes} 

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    if samplingMethod == 1:
        "Only for binary classes. Sample only foreground voxels when present. Sample background voxels only from scans that have NO foreground voxels."
        assert os.path.exists(channels), 'Generating voxel-index for samplig: ERROR: path doesnt exist {}'.format(channels)           
        #Check if previously stored arrays indicating sampling locations:
        exam = channel_mri.split('/')[-2]
        side = channel_mri.split('T1_')[-1][0]
        scan_ID = exam + '_' + side
        if 'BENIGN' in channels:
            benign_scan = True
            voxel_locations_array = CV_FOLDS_ARRAYS_PATH + scan_ID + '_Background_voxel_locations_{}_percentile.npz'.format(percentile_voxel_intensity_sample_benigns)
        else:
            benign_scan = False
            voxel_locations_array = CV_FOLDS_ARRAYS_PATH + scan_ID + '_Foreground_voxel_locations.npz'
            
        if os.path.exists(voxel_locations_array):
            #print('Found previously stored voxel locations..')
            candidate_voxels_for_sampling = np.load(voxel_locations_array, allow_pickle=True)
            candidate_voxels_for_sampling = candidate_voxels_for_sampling[candidate_voxels_for_sampling.keys()[0]]
            
            if (len(range(np.min(candidate_voxels_for_sampling[:,0])-(dpatch[0]/2),np.min(candidate_voxels_for_sampling[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                       (len(range(np.max(candidate_voxels_for_sampling[:,0])-(dpatch[0]/2),np.max(candidate_voxels_for_sampling[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                       (len(range(np.min(candidate_voxels_for_sampling[:,1])-(dpatch[1]/2),np.min(candidate_voxels_for_sampling[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                       (len(range(np.max(candidate_voxels_for_sampling[:,1])-(dpatch[1]/2),np.max(candidate_voxels_for_sampling[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                       (len(range(np.min(candidate_voxels_for_sampling[:,2])-(dpatch[2]/2),np.min(candidate_voxels_for_sampling[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]) or \
                       (len(range(np.max(candidate_voxels_for_sampling[:,2])-(dpatch[2]/2),np.max(candidate_voxels_for_sampling[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]):
                       
                       pdb.set_trace()
            if benign_scan:
                #print('Scan benign. Sampling half from voxel-intensity > {}'.format(percentile_voxel_intensity_sample_benigns))
                # Half from the intensity-based sampling:
                scanVoxels = candidate_voxels_for_sampling[random.sample(range(0,len(candidate_voxels_for_sampling)), 
                                                                         min( len(candidate_voxels_for_sampling), patches_per_subject/2) )].tolist()   
                # Half from random locations:
#                for _ in range(patches_per_subject/2):
#                    x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)+1)) 
#                    y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)+1))
#                    z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)+1))
                for _ in range(patches_per_subject/2):
                    x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2))) 
                    y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)))
                    z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)))
                    scanVoxels.append([x,y,z])      
            else:
                #print('Scan malignant, sampling from labeled region.')
                scanVoxels = candidate_voxels_for_sampling[random.sample(range(0,len(candidate_voxels_for_sampling)), min(len(candidate_voxels_for_sampling),patches_per_subject))]
               
                
        else:
            # No previously stored voxel coordinates for candidate sampling ############
            ############################################################################    
            bV = 0
            fg = 0   
            dontknow = 0;
            if benign_scan:
                # Only getting non-tumor voxels from benign scans:
                if percentile_voxel_intensity_sample_benigns > 0:
                    # Sample high-intensity voxels but also random from the scan.
                    bV = getBodyVoxels(channel_mri, percentile_voxel_intensity_sample_benigns, percentile_normalization)
                    
                    # REMOVE BOUNDARY VOXELS
#                    indBoundary = bV[:,0]<dpatch[0]/2 or bV[:,0]>int(target_shape[0])-dpatch[0]/2-1 or \
#                                  bV[:,1]<dpatch[1]/2 or bV[:,1]>int(target_shape[1])-dpatch[1]/2-1 or \
#                                  bV[:,2]<dpatch[2]/2 or bV[:,2]>int(target_shape[2])-dpatch[2]/2-1
                    try:
                        indBoundary = (bV[:,0]<dpatch[0]/2+dontknow) | (bV[:,0]>int(target_shape[0])-(dpatch[0]/2)-1-dontknow) | \
                                      (bV[:,1]<dpatch[1]/2+dontknow) | (bV[:,1]>int(target_shape[1])-(dpatch[1]/2)-1-dontknow) | \
                                      (bV[:,2]<dpatch[2]/2+dontknow) | (bV[:,2]>int(target_shape[2])-(dpatch[2]/2)-1-dontknow)
                                      
                        bV = bV[~indBoundary]
                        if (len(range(np.min(bV[:,0])-(dpatch[0]/2),np.min(bV[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                           (len(range(np.max(bV[:,0])-(dpatch[0]/2),np.max(bV[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                           (len(range(np.min(bV[:,1])-(dpatch[1]/2),np.min(bV[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                           (len(range(np.max(bV[:,1])-(dpatch[1]/2),np.max(bV[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                           (len(range(np.min(bV[:,2])-(dpatch[2]/2),np.min(bV[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]) or \
                           (len(range(np.max(bV[:,2])-(dpatch[2]/2),np.max(bV[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]):
                           
                           pdb.set_trace()
                        
                        np.savez_compressed(CV_FOLDS_ARRAYS_PATH + scan_ID + '_Background_voxel_locations_{}_percentile'.format(percentile_voxel_intensity_sample_benigns),bV)                    
                        # Half from the intensity-based sampling:
                        scanVoxels = bV[random.sample(range(0,len(bV)), min( len(bV), patches_per_subject/2) )].tolist()   
                        # Half from random locations:
    #                    for _ in range(patches_per_subject/2):
    #                        x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)+1)) 
    #                        y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)+1))
    #                        z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)+1))
                        for _ in range(patches_per_subject/2):
                            x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2))) 
                            y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)))
                            z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)))
                            scanVoxels.append([x,y,z])
                            
                    except:
                        print(channel_mri)
                        sys.exit(0)
                                
                else:
                    scanVoxels = []
#                    for _ in range(patches_per_subject):
#                        x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)+1))  
#                        y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)+1))
#                        z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)+1))
                    for _ in range(patches_per_subject):
                        x = random.choice(range(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)))  
                        y = random.choice(range(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)))
                        z = random.choice(range(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)))
                        scanVoxels.append([x,y,z])
            
            else:
                
                nifti_label = nib.load(channels)
                data_label = np.asanyarray(nifti_label.dataobj)
#                data_label = np.asarray(nifti_label.dataobj)
            
                # Target label contains segmentation of a tumor. Scan is malignant.
                fg = getForegroundBackgroundVoxels(nifti_label, data_label, target_shape) # This function returns only foreground voxels based on labels.
                if len(fg) == 0:
                  print('resize function removed the foreground voxels...')
                  print(channels)
                  print('Target shape = {}'.format(target_shape))
                  print('Original shape = {}'.format(data_label.shape))
                  sys.exit(0)
                
                # REMOVE BOUNDARY VOXELS
                indBoundary = (fg[:,0]<dpatch[0]/2+dontknow) | (fg[:,0]>int(target_shape[0])-(dpatch[0]/2)-1-dontknow) | \
                              (fg[:,1]<dpatch[1]/2+dontknow) | (fg[:,1]>int(target_shape[1])-(dpatch[1]/2)-1-dontknow) | \
                              (fg[:,2]<dpatch[2]/2+dontknow) | (fg[:,2]>int(target_shape[2])-(dpatch[2]/2)-1-dontknow)
                                  
                fg = fg[~indBoundary]
                if (len(range(np.min(fg[:,0])-(dpatch[0]/2),np.min(fg[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                   (len(range(np.max(fg[:,0])-(dpatch[0]/2),np.max(fg[:,0])+(dpatch[0]/2)+dpatch[0]%2)) != dpatch[0]) or \
                   (len(range(np.min(fg[:,1])-(dpatch[1]/2),np.min(fg[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                   (len(range(np.max(fg[:,1])-(dpatch[1]/2),np.max(fg[:,1])+(dpatch[1]/2)+dpatch[1]%2)) != dpatch[1]) or \
                   (len(range(np.min(fg[:,2])-(dpatch[2]/2),np.min(fg[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]) or \
                   (len(range(np.max(fg[:,2])-(dpatch[2]/2),np.max(fg[:,2])+(dpatch[2]/2)+dpatch[2]%2)) != dpatch[2]):
                       
                   pdb.set_trace()
                       
                np.savez_compressed(CV_FOLDS_ARRAYS_PATH + scan_ID + '_Foreground_voxel_locations',fg)
                scanVoxels = fg[random.sample(range(0,len(fg)), min(len(fg),patches_per_subject))]

            del fg
            del bV   
    else:
       raise ValueError('Sampling method not implemented')
         
    #--------------------------------------------------------------------------------------------------------------------------------------------------        
    allVoxelIndexes[subjectIndexes] = scanVoxels
    del scanVoxels
    return allVoxelIndexes    
            
def getSubjectsToSample(channelList, subjectIndexes):
    "Actually returns channel of the subjects to sample"
    fp = open(channelList)
    lines = fp.readlines()
    subjects = [lines[i] for i in subjectIndexes]
    fp.close()
    return subjects

def extractLabels_parallelization(subject_label_channel, voxelCoordinates, output_dpatch, output_shape):
    if len(voxelCoordinates) == 0:
      print('Within extractLabels_parallelization: \nERROR: len(voxelCoordinates) == 0!, subject_label_channel = {}'.format(subject_label_channel))
      sys.exit(0)
    labels = []       
    subject = str(subject_label_channel)[:-1]
    proxy_label = nib.load(subject)
    label_data = np.asarray(proxy_label.dataobj)

    if np.array(label_data.shape != output_shape).any():
       label_data = resize(label_data, order=0, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
    if np.sum(label_data) == 0:
      for j in range(len(voxelCoordinates)):
        labels.append(np.zeros((output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='int8'))
    else:
      for j in range(len(voxelCoordinates)):
        D1,D2,D3 = voxelCoordinates[j]
        
        labels.append(label_data[D1-(output_dpatch[0]//2):D1+(output_dpatch[0]//2)+output_dpatch[0]%2,
                                 D2-(output_dpatch[1]//2):D2+(output_dpatch[1]//2)+output_dpatch[1]%2,
                                 D3-(output_dpatch[2]//2):D3+(output_dpatch[2]//2)+output_dpatch[2]%2])

    proxy_label.uncache()
    
    return labels


def extractLabels(groundTruthChannel_list, subjectIndexes, voxelCoordinates, output_dpatch, shapes):
    #print('extracting labels from ' + str(len(subjectIndexes))+ ' subjects.')    
    subjects = getSubjectsToSample(groundTruthChannel_list,subjectIndexes)
    labels = []       
    for i in range(len(subjects)):
        subject = str(subjects[i])[:-1]
        #print('extracting labels from subject index [{}] with path : {}'.format(subjectIndexes[i],subject))
        proxy_label = nib.load(subject)
        label_data = np.asarray(proxy_label.dataobj)
        # WHERE AM I RESIZING THE SEGMENTATION LABEL???        
        if np.array(label_data.shape != shapes[i]).any():
           label_data = resize(label_data, shapes[i], order=0, preserve_range=True, anti_aliasing=True)        
        #DEBUG
        label_padded = np.pad(label_data,((0,100),(0,100),(0,60)),'constant')  # need to pad for segmentation with huge patches that go outside (only the end - ascending coordinates) boundaries. Scale stays the same, as the origin is not modified. 
        if np.sum(label_data) == 0:
          for j in range(len(voxelCoordinates[i])):
            labels.append(np.zeros((output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='int8'))
        else:
          for j in range(len(voxelCoordinates[i])):
            D1,D2,D3 = voxelCoordinates[i][j]
            #print('Extracting labels from \n subject {} with shape {} and coords {},{},{}'.format(subjects[i], label_data.shape ,D1,D2,D3))
            labels.append(label_padded[D1-output_dpatch[0]//2:D1+(output_dpatch[0]//2)+output_dpatch[0]%2,
                                       D2-output_dpatch[1]//2:D2+(output_dpatch[1]//2)+output_dpatch[1]%2,
                                       D3-output_dpatch[2]//2:D3+(output_dpatch[2]//2)+output_dpatch[2]%2])
            #if len(labels[-1])==0:
            #  labels[-1] = np.zeros((9,9),dtype='int8')
        proxy_label.uncache()
        del label_data
    return labels

#shapes = [shape]
#voxelCoordinates = [voxelCoordinates]
   
def extractCoordinates(shapes, voxelCoordinates, output_dpatch):
    """ Given a list of voxel coordinates, it returns the absolute location coordinates for a given patch size (output 1x9x9) """
    #print('extracting coordinates from ' + str(len(subjectIndexes))+ ' subjects.')
    #subjects = getSubjectsToSample(channel, subjectIndexes)
    
    all_coordinates = []
    for i in range(len(shapes)):
        #subject = str(subjects[i])[:-1]
        #img = nib.load(subject)
        img_shape = shapes[i]
        for j in range(len(voxelCoordinates[i])):     
            D1,D2,D3 = voxelCoordinates[i][j]
            #all_coordinates.append(get_Coordinates_from_target_patch(img.shape,D1,D2,D3))                 
            all_coordinates.append(get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch))                    

        #img.uncache()
    return np.array(all_coordinates)    


def get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch) :

    x_ = range(D1-(output_dpatch[0]//2),D1+((output_dpatch[0]//2)+output_dpatch[0]%2))
    y_ = range(D2-(output_dpatch[1]//2),D2+((output_dpatch[1]//2)+output_dpatch[1]%2))
    z_ = range(D3-(output_dpatch[2]//2),D3+((output_dpatch[2]//2)+output_dpatch[2]%2))
    
    x_norm = np.array(x_)/float(img_shape[0])  
    y_norm = np.array(y_)/float(img_shape[1])  
    z_norm = np.array(z_)/float(img_shape[2])  
    
    x, y, z = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')    
    coords = np.stack([x,y,z], axis=-1)
    return coords
    
       
def get_patches_per_subject( n_patches, n_subjects):
    patches_per_subject = [n_patches/n_subjects]*n_subjects
    randomAdd = random.sample(range(0,len(patches_per_subject)),k=n_patches%n_subjects)
    randomAdd.sort()
    for index in randomAdd:
        patches_per_subject[index] = patches_per_subject[index] + 1
    return patches_per_subject

def extractImagePatch_parallelization(channel, subjectIndex, subject_channel_voxelCoordinates, output_shape, dpatch, percentile_normalization, preprocess_image_data=True,fullSegmentationPhase=False):   
    subject_channel = getSubjectsToSample(channel, [subjectIndex])
    n_patches = len(subject_channel_voxelCoordinates)
    subject = str(subject_channel[0])[:-1]
    proxy_img = nib.load(subject) #added [:-1] and removed
    
#    tt = time.time()
    img_data = np.asarray(proxy_img.dataobj)
    
    # Across-exam normalization below # ANDY 2020-08-25
    if 'jhuData' in subject:
        
        if 'axial_02' in subject:
            img_data = img_data/20.35
        elif 'slope1' in subject:
            img_data = img_data/0.14 #0.55
        elif 'slope2' in subject:
            img_data = img_data/0.07
            
    else:
        
        if 'axial_02' in subject:
            img_data = img_data/25.17
        elif 'slope1' in subject:
            img_data = img_data/0.18
        elif 'slope2' in subject:
            img_data = img_data/0.05
            
#    img_data = np.transpose(img_data, (2,0,1))
#    # PERMUTE ON THE FLY # ANDY 2020-02-16
    
#     if preprocess_image_data:   
    if np.array(img_data.shape != output_shape).any():
      print('Resizing training data: \nInput_shape = {}, \nOutput_shape = {}. \nSubject = {}'.format(img_data.shape, output_shape, subject))
      img_data = resize(img_data, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

    if fullSegmentationPhase:      
        padding_border = np.max(dpatch)#np.max(dpatch)/2 + 10#550
#        proxy_img = nib.load(subject)
#        img_data = np.asarray(proxy_img.dataobj)
        img_data = np.pad(img_data, padding_border,'reflect')
    
    vol = np.zeros((n_patches,dpatch[0],dpatch[1],dpatch[2]),dtype='float32') 
    for j in range(n_patches):      
        D1,D2,D3 = subject_channel_voxelCoordinates[j]
        
        if not fullSegmentationPhase:
               
            try:

                vol[j,:,:,:] = img_data[D1-(dpatch[0]//2):D1+(dpatch[0]//2)+dpatch[0]%2,
                                        D2-(dpatch[1]//2):D2+(dpatch[1]//2)+dpatch[1]%2,
                                        D3-(dpatch[2]//2):D3+(dpatch[2]//2)+dpatch[2]%2]

            except:
                print('Failed at {} in image {} of shape {}'.format([D1,D2,D3],subject_channel,img_data.shape))
                sys.exit(0)
            
        else:
                
            D1 = D1 + padding_border#dpatch[0]/2
            D2 = D2 + padding_border#dpatch[1]/2
            D3 = D3 + padding_border#dpatch[2]/2
            
            vol[j,:,:,:] = img_data[D1-(dpatch[0]//2):D1+(dpatch[0]//2)+dpatch[0]%2,
                                    D2-(dpatch[1]//2):D2+(dpatch[1]//2)+dpatch[1]%2,
                                    D3-(dpatch[2]//2):D3+(dpatch[2]//2)+dpatch[2]%2]
#    print(time.time() - tt)
#    proxy_img.uncache()
    # del img_data
    # del img_data_padded
    return vol


#def sampleTrainData_daemon(return_dict, procnum, resolution, trainChannels,CV_FOLDS_ARRAYS_PATH, trainLabels, TPM_channel, n_patches, n_subjects, dpatch, output_classes, samplingMethod, use_coordinates, proportion_malignants_to_sample, percentile_voxel_intensity_sample_benigns, data_augmentation, proportion_to_flip, percentile_normalization, model_patch_reduction, model_crop, balanced_sample_subjects=True, verbose=False, debug=False, using_unet=True):
def sampleTrainData_daemon(procnum, resolution, trainChannels,CV_FOLDS_ARRAYS_PATH, trainLabels, TPM_channel, n_patches, n_subjects, dpatch, output_classes, samplingMethod, use_coordinates, proportion_malignants_to_sample, percentile_voxel_intensity_sample_benigns, data_augmentation, proportion_to_flip, percentile_normalization, model_patch_reduction, model_crop, balanced_sample_subjects=True, verbose=False, debug=False, using_unet=True):
    #TODO: simplify this function.
    #TODO: remove unneeded arguments.
    num_channels = len(trainChannels)
    output_dpatch = dpatch[0] - model_patch_reduction[0], dpatch[1] - model_patch_reduction[1], dpatch[2] - model_patch_reduction[2]
    patches_per_subject = get_patches_per_subject( n_patches, n_subjects)    
    labelsFile = open(trainLabels).readlines()    
    total_subjects = len(labelsFile)

    if balanced_sample_subjects:     
      proportion_malignants = int(np.ceil(n_subjects*proportion_malignants_to_sample))
      malignant_subjects_index = [labelsFile.index(x) for x in labelsFile if not 'BENIGN' in x]
      benign_subjects_index = list(set(range(total_subjects)) - set(malignant_subjects_index))
      subjectIndexes = random.sample(malignant_subjects_index, min(len(malignant_subjects_index), proportion_malignants))
      print('{} : sampling {} malignants from partition'.format(procnum,len(subjectIndexes)))
      try:
        subjectIndexes.extend(random.sample(benign_subjects_index, n_subjects - len(subjectIndexes)))
      except:
        # if not enough or only malignants in set.
        subjectIndexes.extend(random.sample(malignant_subjects_index, n_subjects - len(subjectIndexes)))
      random.shuffle(subjectIndexes)
    else:
      print('{} : Extracting data from randomly selected subjects.. [breast mask model]'.format(procnum))  
      subjectIndexes = generateRandomIndexesSubjects(n_subjects, total_subjects)  
    
    #------------- Parallelization ----------------
    channel_mri = getSubjectChannels(subjectIndexes, trainChannels[0]) 
    #GET_SHAPES_INPUT = zip(channel_mri, [CV_FOLDS_SHAPES_PATH]*len(channel_mri))
    
#    pool = Pool(min(multiprocessing.cpu_count()/12,19))#mp.cpu_count() -1)
    time1 = time.time()
#    shapes = pool.map(getSubjectShapes_parallelization, zip(channel_mri,[resolution]*len(channel_mri)))
    shapes = []
    for i in range(0,len(channel_mri)):
        shapes.append(getSubjectShapes_parallelization(channel_mri[i],resolution))
    print('{} : Getting scan shapes took {} s'.format(procnum, round(time.time() - time1,2)))

    ############ Generating Voxel Coordinates For training ##############    
    print('{} : ------------ Generating List of Voxel Indexes for sampling  ------------'.format(procnum))
    #------------ Parallelization --------------------
    channels = getSubjectChannels(subjectIndexes, trainLabels)
#    channel_mri = getSubjectChannels(subjectIndexes, trainChannels[0]) 
    
    # INPUT SHOULD BE OUTPUT PATCH, NOT DPATCH!!! WE PUT CONSTRAIN ON BORDER OF OUTPUT NOT INPUT!! 
    
#    DATA_INPUT_WRAPPER = zip(subjectIndexes, shapes, patches_per_subject, channels, channel_mri, [dpatch] * len(patches_per_subject), [n_patches]*len(patches_per_subject), [samplingMethod]*len(patches_per_subject), [output_classes]*len(patches_per_subject), [percentile_voxel_intensity_sample_benigns]*len(patches_per_subject), [percentile_normalization]*len(patches_per_subject), [CV_FOLDS_ARRAYS_PATH]*len(patches_per_subject))
#    DATA_INPUT_WRAPPER = list(DATA_INPUT_WRAPPER)
#    #pool = Pool(multiprocessing.cpu_count() - 1)#mp.cpu_count() -1)
    time1 = time.time()
#    try:      
#      voxelCoordinates_full = pool.map(generateVoxelIndexes_wrapper_parallelization, DATA_INPUT_WRAPPER)
    voxelCoordinates_full = []
    for i in range(0,len(subjectIndexes)):
        voxelCoordinates_full.append(generateVoxelIndexes_parallel(subjectIndexes[i],CV_FOLDS_ARRAYS_PATH, shapes[i], patches_per_subject[i], dpatch, n_patches, channels[i], channel_mri[i], samplingMethod, output_classes, percentile_voxel_intensity_sample_benigns,percentile_normalization, allForegroundVoxels = "", verbose=False))
#    except IndexError:
#      print('...IndexError ')
#      print(DATA_INPUT_WRAPPER)
    print('{} : Generating List of Voxel Indexes for sampling took {} s'.format(procnum, round(time.time() - time1,2)))
    #pool.close()
    #pool.join()  
    subjectIndexes_check = []
    voxelCoordinates = []
    for i in range(len(voxelCoordinates_full)):
      subjectIndexes_check.append(list(voxelCoordinates_full[i].keys())[0])
      voxelCoordinates.extend(voxelCoordinates_full[i].values())
    assert subjectIndexes_check == subjectIndexes, 'Subject Indexes got out of order through multiprocessing...'
    del subjectIndexes_check
    del voxelCoordinates_full
    
    real_n_patches = 0
    for i in range(len(voxelCoordinates)):
        if len(voxelCoordinates[i]) == 0:
          print('Empty voxelCoordinates for i = {}, channel_mri[i] = {}, \n\n DATA_INPUT_WRAPPER[i] = {}, \n\n patches_per_subject = {},\n patches_per_subject[i] = {}'.format(i, channel_mri[i], DATA_INPUT_WRAPPER[i], patches_per_subject, patches_per_subject[i]))
          sys.exit(0)
        real_n_patches += len(voxelCoordinates[i])        
    print('{} : ------  Extracting {} image patches from {} subjects, for each of {} channels --------'.format(procnum, real_n_patches,len(voxelCoordinates), len(trainChannels) ))

    ############## Parallelization Extract Image Patches ####################
    patches = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')       
    for i in range(len(trainChannels)):
        print('{} : Extracting image patches from channel: {}'.format(procnum, trainChannels[i]))
#        DATA_INPUT_EXTRACT_IMAGE_PATCH = zip(subjectIndexes, [trainChannels[i]]*len(subjectIndexes), [dpatch] * len(subjectIndexes), voxelCoordinates, shapes, [percentile_normalization]*len(subjectIndexes))
#        DATA_INPUT_EXTRACT_IMAGE_PATCH = list(DATA_INPUT_EXTRACT_IMAGE_PATCH)
#        #pool = Pool(multiprocessing.cpu_count() -1) #-1 )
        time1 = time.time()
#        channel_patches = pool.map(extractImagePatch_parallelization_wrapper, DATA_INPUT_EXTRACT_IMAGE_PATCH)
        channel_patches = []
        for j in range(0,len(subjectIndexes)):
            #print('reading '+ str(j) + ' out of ' +str(len(subjectIndexes)) +'...')
            channel_patches.append(extractImagePatch_parallelization(trainChannels[i], subjectIndexes[j], voxelCoordinates[j], shapes[j], dpatch, percentile_normalization))
        print('{} : Extracting image patches took {} s'.format(procnum, round(time.time() - time1,2))) 
        
        try:
            channel_patches_flattened = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2]))
        except:
            print('memory error at {}'.format(subjectIndexes))
            sys.exit(0)
        start = 0
        for ii in range(len(channel_patches)):
          channel_patches_flattened[start:start+len(channel_patches[ii])] = channel_patches[ii]
          start = start+len(channel_patches[ii])        
        del channel_patches
        patches[:,:,:,:,i] = channel_patches_flattened
    print('{} : ------  Extracting {} target-label patches from {} subjects --------'.format(procnum, real_n_patches,len(voxelCoordinates) ))

    ############# Parallelization Extract Label Patches ###################### 
    subjects_label_channels = getSubjectsToSample(trainLabels,subjectIndexes)          
#    DATA_INPUT_EXTRACT_LABELS_PATCH = zip(subjects_label_channels, voxelCoordinates, [output_dpatch] * len(subjectIndexes), shapes)
#    DATA_INPUT_EXTRACT_LABELS_PATCH = list(DATA_INPUT_EXTRACT_LABELS_PATCH)
#    #pool = Pool(multiprocessing.cpu_count() -1) #-1 )
    time1 = time.time()
#    labels_list_unflattened = pool.map(extractLabels_parallelization_wrapper, DATA_INPUT_EXTRACT_LABELS_PATCH)
    labels_list_unflattened = []
    for i in range(0,len(subjects_label_channels)):
        labels_list_unflattened.append(extractLabels_parallelization(subjects_label_channels[i], voxelCoordinates[i], output_dpatch, shapes[i]))
    print('{} : Extracting target label patches took {} s'.format(procnum, round(time.time() - time1,2)))   
    #pool.close()
    #pool.join() 
    labels = np.zeros(([real_n_patches] + list(output_dpatch) ), dtype='int8')
    start = 0
    try:
      for ii in range(len(labels_list_unflattened)):
        labels[start:start+len(labels_list_unflattened[ii])] = labels_list_unflattened[ii]
        start = start+len(labels_list_unflattened[ii])        
    except ValueError:
      print('ValueError... ii = {}, start={}, len(labels_list_unflattened[ii]) = {},  subjects_label_channels[ii]: {}'.format(ii, start,len(labels_list_unflattened[ii]), subjects_label_channels[ii] ))
    del labels_list_unflattened    

    labels = np.array(labels,dtype='int8')
    labels_list = np.array(labels)
    labels = np.array(to_categorical(labels.astype(int),output_classes),dtype='int8')
    # FOR AXIAL DATA
    if len(np.shape(labels))<5:
       labels = np.reshape(labels,newshape=[real_n_patches,output_dpatch[0],output_dpatch[1],1,output_classes])

    #if(samplingMethod == 2):
    #    patches = patches[0:len(labels)]  # when using equal sampling (samplingMethod 2), because some classes have very few voxels in a head, there are fewer patches as intended. Patches is initialized as the maximamum value, so needs to get cut to match labels.

    ############ Location Coordinates ####################
    if use_coordinates:
      all_coordinates = extractCoordinates(shapes, voxelCoordinates, output_dpatch)
      if debug:
        y_coords = all_coordinates[:,2,:,:]
        plt.imshow(y_coords[1])
        center_y = y_coords[:,5,1]
        plt.hist(center_y,  200)
        plt.xlabel('Normalized Y coordinate')

    else:
      all_coordinates = []
      
    ############ TPM Patches #############################  
    if len(TPM_channel) > 0:
      print('{} : ------  Extracting {} TPM patches from {} subjects --------'.format(procnum, real_n_patches,len(voxelCoordinates) ))      
      TPM_INPUT_DATA = zip([TPM_channel]*len(subjectIndexes), subjectIndexes, voxelCoordinates, [output_dpatch] * len(subjectIndexes), shapes)
      TPM_INPUT_DATA = list(TPM_INPUT_DATA)
      time1 = time.time()
      TPM_patches_unflattened = pool.map(extract_TPM_patches_parallelization_wrapper, TPM_INPUT_DATA)
      print('{} : Extracting TPM patches took {} s'.format(procnum, round(time.time() - time1,2))) 
      TPM_patches = np.zeros(([real_n_patches] + list(output_dpatch)),dtype='float32')
      start = 0
      for ii in range(len(TPM_patches_unflattened)):
        TPM_patches[start:start+len(TPM_patches_unflattened[ii])] = TPM_patches_unflattened[ii]
        start = start+len(TPM_patches_unflattened[ii])        
      del TPM_patches_unflattened              
         
    else:
      TPM_patches = []  
      
#    pool.close()
#    pool.join() 
      
    if debug:
#        patches.shape
        labels_img = np.array(labels_list,dtype='int8')
        for display_index in range(50,60,1):
            plt.figure(figsize=(12,8))
            plt.subplot(221)
            plt.imshow(patches[display_index,:,:,9,0], cmap='gray')
            plt.title('T1post'+str(voxelCoordinates[5][display_index-50]))
            plt.subplot(222)
            plt.imshow(patches[display_index,:,:,9,1], cmap='gray')
            plt.title('slope1')
            plt.subplot(223)
            plt.imshow(patches[display_index,:,:,9,2], cmap='gray')
            plt.title('slope2')
            plt.subplot(224)
            plt.imshow(labels_img[display_index,:,:,0], cmap='gray')
            plt.title('Target label')
            plt.savefig('/home/andy/projects/lukasSegmenter/biomedical_segmenter/training_sessions/lukasSegmenter/One_Patch_example_{}.png'.format(display_index))
    
    if data_augmentation:
        print('{} : Data augmentation: Randomly flipping {}% patches..'.format(procnum, proportion_to_flip*100))
        patches, labels, TPM_patches = flip_random(patches, labels, TPM_patches, all_coordinates, proportion_to_flip)
      
    print('{} : Checking for NaN in image patches..'.format(procnum))    
    if np.any(np.isnan(patches)):
        print('{} : nan found in the input data batch for training..'.format(procnum))
        print(patches[np.isnan(patches)].shape)
        patches[np.isnan(patches)] = 0.0
    assert not np.any(np.isnan(patches)), 'STILL NANs!'          
    if np.any(~ np.isfinite(patches)):
        patches[~ np.isfinite(patches)] = 0.0
    assert np.all(np.isfinite(patches)), 'STILL Non-Finite Values!'
    #print('{} : Number of class 0 samples in whole batch: {}'.format(procnum, np.sum(labels[:,0])))
    #print('{} : Number of class 1 samples in whole batch: {}'.format(procnum, np.sum(labels[:,1])))
    #### SHUFFLE ####
    print('{} : Shuffling data..'.format(procnum))
    shuffleOrder = np.arange(patches.shape[0])
    np.random.shuffle(shuffleOrder)
    patches = patches[shuffleOrder]
    labels = labels[shuffleOrder]  
    if len(all_coordinates) > 0:
        all_coordinates = all_coordinates[shuffleOrder]
    if len(TPM_patches) > 0:
        TPM_patches = TPM_patches[shuffleOrder]        
    #--------------------------------------------------------------------------
    # Preprocess patches according to model needs... (BreastMask vs Segmenter...)
    # Resize Context Patch
    if using_unet:
        context = []
    else:
        context = np.array(patches[:,:,:,:,0],'float')
        context = resize(image=context, order=1, 
                             output_shape=(context.shape[0],context.shape[1],context.shape[2]/3,context.shape[3]/3), 
                             anti_aliasing=True, preserve_range=True )    
    # Crop Detail Patch
    if model_crop > 2:
        patches = np.array(patches[:,:,model_crop/2:-model_crop/2,model_crop/2:-model_crop/2,:])
    print('{} : ------- Finished data sampling -------'.format(procnum))    
#    return_dict[procnum] = context, patches, labels, all_coordinates, TPM_patches
    return context, patches, labels, all_coordinates, TPM_patches

def sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, output_dpatch, shape, use_coordinates):
   #TODO: remove unused parameters (output_classes)
    xend = output_dpatch[0] * int(round(float(shape[0])/output_dpatch[0] + 0.5)) 
    if shape[1] == output_dpatch[1]:
        yend = output_dpatch[1]
    else:
        yend = output_dpatch[1] * int(round(float(shape[1])/output_dpatch[1] + 0.5)) 
    if shape[2] == output_dpatch[2]:
        zend = output_dpatch[2]
    else:           
        zend = output_dpatch[2] * int(round(float(shape[2])/output_dpatch[2] + 0.5))
    voxelCoordinates = []
    # Remember in python the end is not included! Last voxel will be the prior-to-last in list.
    # It is ok if the center voxel is outside the image, PROPER padding will take care of that (if outside the image size, then we need larger than dpatch/2 padding)
    
    for x in range(output_dpatch[0]//2,xend,output_dpatch[0]): 
        for y in range(output_dpatch[1]//2,yend,output_dpatch[1]):
            for z in range(output_dpatch[2]//2,zend,output_dpatch[2]):
                voxelCoordinates.append([x,y,z])
    
    if len(TPM_channel) > 0:
      TPM_patches = extract_TPM_patches(TPM_channel, subjectIndex, [voxelCoordinates], output_dpatch, [shape])
    else:
      TPM_patches = []
    if len(testLabels) > 0:
      labels = np.array(extractLabels(testLabels, subjectIndex, [voxelCoordinates], output_dpatch,[shape]))
      labels = to_categorical(labels.astype(int),output_classes)
    else:
      labels = []
    if use_coordinates:
      spatial_coordinates = extractCoordinates([shape], [voxelCoordinates], output_dpatch) 
    else:
      spatial_coordinates = []
    #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
    return TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape        



def getForegroundBackgroundVoxels(nifti_label, data_label, target_shape):
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    shape = nifti_label.shape
    if np.array(shape != target_shape).any():
      data = resize(data_label, order=0, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
    else:
      data = data_label
    if np.sum(data) == 0:
      data = resize(data_label,order=1, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
      data[data > 0] = 1
      
#    data = data_label
    nifti_label.uncache()    
    foregroundVoxels = np.argwhere(data>0)
    return foregroundVoxels
      
def getBodyVoxels(channel, percentile_voxel_intensity_sample_benigns, percentile_normalization):
    '''Get vector of voxel coordinates for all voxel values > 0'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(channel)
    data = np.asanyarray(img.dataobj)
    
    # Across-exam normalization below # ANDY 2020-08-25
    if 'jhuData' in channel:
        
        if 'axial_02' in channel:
            data = data/20.35
        elif 'slope1' in channel:
            data = data/0.14 #0.55
        elif 'slope2' in channel:
            data = data/0.07
            
    else:
        
        if 'axial_02' in channel:
            data = data/25.17
        elif 'slope1' in channel:
            data = data/0.18
        elif 'slope2' in channel:
            data = data/0.05
        
    res = img.header['pixdim'][1:4]
    shape = img.shape 
    if res[0] > 1.0:    
      target_res = [res[0]/2.,res[1]/2.,res[2]]
      out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
      data = resize(data,order=1, output_shape=out_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        
#    if percentile_normalization:
#      data = percentile95_normalizeMRI(data)
#    else:  
#      data = normalizeMRI(data)
#    if np.any(np.isnan(data)):
#      print('NaN found on getBodyVoxels Function')
#      sys.exit(0)
    img.uncache()    
    bodyVoxels = np.argwhere(data > np.percentile(data, percentile_voxel_intensity_sample_benigns))
    return bodyVoxels

####################################### METRIC FUNCTIONS #################################################

#def generalized_dice_completeImages(img1,img2):
#    assert img1.shape == img2.shape, 'Images of different size!'
#    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
#    classes = np.array(np.unique(img1), dtype='int8')   
#    if len(classes) < len(np.array(np.unique(img2), dtype='int8')   ):
#      classes = np.array(np.unique(img2), dtype='int8')   
#    dice = []
#    for i in classes:
#        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
#    return np.sum(dice)/len(classes), [round(x,2) for x in dice]

#def dice_coef_multilabel_bin0(y_true, y_pred):
#    index = 0
#    dice = dice_coef(y_true[:,:,:,:,index], K.round(y_pred[:,:,:,:,index]))
#    return dice

#def dice_coef_multilabel_bin1(y_true, y_pred):
#    index = 1
#    dice = dice_coef(y_true[:,:,:,:,index], K.round(y_pred[:,:,:,:,index]))
#    return dice

################################## DOCUMENTATION FUNCTIONS ################################################

def my_logger(string, logfile, print_out=True):
    f = open(logfile,'a')
    f.write('\n' + str(string))
    f.close()
    if print_out:
        print(string)
    

def start_training_session_logger(logfile,threshold_EARLY_STOP, TPM_channel,saveSegmentation,path_to_model,model,dropout, trainChannels, trainLabels, validationChannels, validationLabels, testChannels, testLabels, num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches):
    #Called within train_test_model
    #Logs all the parameters of the training session
    my_logger('#######################################  NEW TRAINING SESSION  #######################################', logfile)    
    my_logger(trainChannels, logfile)
    my_logger(trainLabels, logfile)
    my_logger(validationChannels, logfile)        
    my_logger(validationLabels, logfile)  
    my_logger(testChannels, logfile) 
    my_logger(testLabels, logfile)
    my_logger('TPM channel (if given):', logfile)
    my_logger(TPM_channel, logfile)
    my_logger('Session parameters: ', logfile)
    my_logger('[num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches]', logfile)
    my_logger([num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches], logfile)
    my_logger('Dropout for last two fully connected layers: ' + str(dropout), logfile)
    my_logger('Model loss function: ' + str(model.loss), logfile)
    my_logger('Model number of parameters: ' + str(model.count_params()), logfile)
    my_logger('Optimizer used: ' +  str(model.optimizer.from_config), logfile)
    my_logger('Optimizer parameters: ' + str(model.optimizer.get_config()), logfile)
    my_logger('Save full head segmentation of subjects: ' + str(saveSegmentation), logfile)
    my_logger('EARLY STOP Threshold last 3 epochs: ' + str(threshold_EARLY_STOP), logfile)

class LossHistory_multiDice2(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.dice = []
        self.metrics = []

    def on_train_batch_end(self, batch, logs={}):
        self.dice = []
        self.losses.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef_multilabel_bin0'))
        self.dice.append(logs.get('dice_coef_multilabel_bin1'))
        self.metrics.append(self.dice)

    def on_train_batch_begin(self, batch, logs={}):
        #Added due to error when missing
        return

        
        
################################### SEGMENTATION FUNCTIONS ##################################################

def fullSegmentation(wd, penalty_MATRIX, resolution, OUTPUT_PATH, TPM_channel, dice_compare, dsc, smooth_dice_scores,foreground_percent_list, model, testChannels, testLabels, subjectIndex, output_classes, segmentation_dpatch, size_minibatches,output_probability, use_coordinates, percentile_normalization, model_patch_reduction, model_crop, epoch, using_breastMaskModel=False, MASK_BREAST=False, using_Unet=False, using_unet_breastMask=False):    
    #TODO: Remove unused parameters
    #TODO: Add comments
    output_dpatch = segmentation_dpatch[0] - model_patch_reduction[0], segmentation_dpatch[1] - model_patch_reduction[1], segmentation_dpatch[2] - model_patch_reduction[2]
    if len(testLabels) == 0:
        dice_compare = False         

    subjectIndex = [subjectIndex]
    num_channels = len(testChannels)
    firstChannelFile = open(testChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    print('SEGMENTATION : Segmenting subject: ' + str(subID))  
    segmentationName =  subID + '_epoch' + str(epoch)
    if len(OUTPUT_PATH) > 0:
        output = OUTPUT_PATH + '/' + segmentationName + '.nii.gz' 
    else:
        output = wd + '/predictions/' + segmentationName + '.nii.gz'
    if os.path.exists(output):
      print('SEGMENTATION : Segmentation already done. Skip')
#      return [None]*5
      return [None, output, None, None, None]
    
    firstChannelFile.close()      
    proxy_img = nib.load(subjectGTchannel) #Added [:-1] to fix bug and removed
    shape = proxy_img.shape
    affine = proxy_img.affine      
    res = proxy_img.header['pixdim'][1:4]

    if resolution == 'high':
        if res[0] > 1.0:    
          target_res = [res[0]/2.,res[1]/2.,res[2]]
          shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
        else:
          target_res = res
    elif resolution == 'low':
        if res[0] < 0.5:    
          target_res = [res[0]*2.,res[1]*2.,res[2]]
          shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
        else:
          target_res = res 
#    target_res = res         
          
    print('SEGMENTATION : Sampling data..')  
    TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape = sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, 
                                                                                       output_dpatch, shape, use_coordinates)    
    affine = np.diag(list(target_res) + [0])        
    n_minibatches = 0 # min(0,len(voxelCoordinates)/size_minibatches) 
    total_number_of_patches = (len(voxelCoordinates)-n_minibatches*size_minibatches)  
    
    #########################################################################
    print('SEGMENTATION : Extracting {} image patches..'.format(total_number_of_patches))
    patches = np.zeros((total_number_of_patches,segmentation_dpatch[0],segmentation_dpatch[1],segmentation_dpatch[2],num_channels),dtype='float32')
    for i in range(len(testChannels)):
        patches[:,:,:,:,i] = extractImagePatch_parallelization(testChannels[i], subjectIndex[0], voxelCoordinates, shape, segmentation_dpatch, percentile_normalization, fullSegmentationPhase=True)    

    print('SEGMENTATION : Finished sampling data into patches of shape ' + str(patches.shape))
#    if debug_coords:
#        patches = np.ones(patches.shape)
#        spatial_coordinates = np.zeros(spatial_coordinates.shape)
    
    INPUT_DATA = []  
    
    # NEED TO ADAPT FOR BREAST MASK MODEL: Inputs: Context (no resizing, 13,75,75) and spatial coordinates.
    if using_breastMaskModel:
        INPUT_DATA.append(patches[:,:,:,:,0].reshape(patches[:,:,:,:,0].shape + (1,)))  
        INPUT_DATA.append(spatial_coordinates)    
  
    elif using_Unet:  # This means the model is the U-Net
        if using_unet_breastMask:
            patches = patches[:,:,:,:,0]
            patches = patches.reshape(patches.shape + (1,))
        INPUT_DATA.append(patches)
#        if len(TPM_patches) > 0:
#            INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
        if len(spatial_coordinates) > 0:
            INPUT_DATA.append(spatial_coordinates)            
            
        
    else:
        # Context
        context = np.array(patches[:,:,:,:,0],'float')
        context = resize(image=context, order=1, 
                             output_shape=(context.shape[0],context.shape[1],context.shape[2]/3,context.shape[3]/3), 
                             anti_aliasing=True, preserve_range=True )
        INPUT_DATA.append(context.reshape(context.shape + (1,)))        
        
        for jj in range(patches.shape[-1]):
            INPUT_DATA.append(patches[:,:,model_crop/2:-model_crop/2,model_crop/2:-model_crop/2,jj].reshape(patches[:,:,model_crop/2:-model_crop/2,model_crop/2:-model_crop/2,jj].shape + (1,)))  
        if len(TPM_patches) > 0:
            INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
        if len(spatial_coordinates) > 0:
            INPUT_DATA.append(spatial_coordinates)    
    
    print("SEGMENTATION : Finished preprocessing data for segmentation.")
    #########################################################################
      
#    prediction = model.predict(INPUT_DATA, verbose=1, batch_size=size_minibatches)
    prediction = model.predict(patches, verbose=1, batch_size=size_minibatches)
     
    ##########  Output binary ############          
    indexes = []
    print('Prediction shape: {}'.format(prediction.shape))
    class_pred = np.argmax(prediction, axis=4)
    indexes.extend(class_pred)     
           
    head = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
    i = 0
    for x,y,z in voxelCoordinates:
        patch_shape = head[x-output_dpatch[0]//2:min(x+(output_dpatch[0]//2+output_dpatch[0]%2), shape[0]),
                           y-output_dpatch[1]//2:min(y+(output_dpatch[1]//2+output_dpatch[1]%2), shape[1]),
                           z-output_dpatch[2]//2:min(z+(output_dpatch[2]//2+output_dpatch[2]%2), shape[2])].shape
        #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
        head[x-output_dpatch[0]//2:min(x+(output_dpatch[0]//2+output_dpatch[0]%2), shape[0]),
             y-output_dpatch[1]//2:min(y+(output_dpatch[1]//2+output_dpatch[1]%2), shape[1]),
             z-output_dpatch[2]//2:min(z+(output_dpatch[2]//2+output_dpatch[2]%2), shape[2])] = np.array(indexes[i])[0:patch_shape[0], 
                                                                                                                   0:patch_shape[1],
                                                                                                                   0:patch_shape[2]]
        i = i+1
    #img_binary = nib.Nifti1Image(head, affine)
    img_binary = head

#    if(saveSegmentation):
#        nib.save(img_binary, output)
#        my_logger('Saved segmentation of subject at: ' + output, logfile)

    ##########  Output probabilities ############
    if output_probability:
        indexes = []        
        class_pred = prediction[:,:,:,:,1]
        indexes.extend(class_pred)     
               
        head = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
        i = 0
        for x,y,z in voxelCoordinates:
            patch_shape = head[x-output_dpatch[0]//2:min(x+(output_dpatch[0]//2+output_dpatch[0]%2), shape[0]),
                               y-output_dpatch[1]//2:min(y+(output_dpatch[1]//2+output_dpatch[1]%2), shape[1]),
                               z-output_dpatch[2]//2:min(z+(output_dpatch[2]//2+output_dpatch[2]%2), shape[2])].shape
            #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
            head[x-output_dpatch[0]//2:min(x+(output_dpatch[0]//2+output_dpatch[0]%2), shape[0]),
                 y-output_dpatch[1]//2:min(y+(output_dpatch[1]//2+output_dpatch[1]%2), shape[1]),
                 z-output_dpatch[2]//2:min(z+(output_dpatch[2]//2+output_dpatch[2]%2), shape[2])] = np.array(indexes[i])[0:patch_shape[0], 
                                                                                                                       0:patch_shape[1],
                                                                                                                       0:patch_shape[2]]
            i = i+1
        #img_probs = nib.Nifti1Image(head, affine)
        img_probs = head

    foreground_percent = 0 
    score_smooth = 0    
    if dice_compare:
      LABEL_CHANNEL = open(testLabels).readlines()[subjectIndex[0]][:-1]
      print('SEGMENTATION : Comparing with Ground-Truth label: {}'.format(LABEL_CHANNEL))
      if 'BENIGN' in LABEL_CHANNEL:
        dice_compare = False
        label_data = np.zeros((img_binary.shape))
        foreground_percent = np.sum(img_binary)/float(head.size)    
        foreground_percent_list.append(foreground_percent)    
      elif np.sum(np.asanyarray(nib.load(LABEL_CHANNEL).dataobj)) == 0:   
        print('SEGMENTATION : Empty label!')
        label_data = np.zeros((img_binary.shape))
      else:
        label_data = np.asanyarray(nib.load(LABEL_CHANNEL).dataobj)    
        print('SEGMENTATION : Label shape = {}'.format(label_data.shape))
        print('SEGMENTATION : Segmentation shape = {}'.format(shape))
        print('label data sum = {}'.format(np.sum(label_data)))
        
        if label_data.shape != shape:  
          print('SEGMENTATION : Resizing.')
          label_data = resize(label_data, output_shape=shape, preserve_range=True, anti_aliasing=True, order=0)
          print('SEGMENTATION: np.sum after resizing: {}'.format(np.sum(label_data)))
          
        if np.any(np.isnan(label_data)):
          label_data[np.isnan(label_data)] = 0

        def dice_class1(img1, img2, THRESHOLD=0.5):
            i = 1
            j = 0
            #img2a = np.zeros_like(img2)
            img2[img2 > THRESHOLD] = 1.
            img2[img2 <= THRESHOLD] = 0.
            return [2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)), 2*np.sum(np.multiply(img1==j,img2==j))/float(np.sum(img1==j)+np.sum(img2==j))]
        
        #Adjusted to fix error/bug
        try:
            sl = np.argwhere(label_data > 0)[0][2]
        except:
            sl = np.argwhere(label_data > 0)[2]
        #TODO - Ensure dice calculation operating correctly
        score = dice_class1(label_data[:,:,sl], img_binary[:,:,sl])
        #score = dice_class1(label_data[:,:,sl], img_binary[:,:,sl], np.max(img_binary)*.95)
        score_smooth = dice_class1(img_binary[:,:,sl], label_data[:,:,sl])[0]
        
        
        dsc.append(score[0])
        smooth_dice_scores.append(score_smooth)
        print(dsc[-1])
        print('per class dice score: {}'.format(score))
        print('mean DCS so far:' + str(np.mean(dsc)))
        print('smooth_dice score: {}'.format(score_smooth))
        print('mean SMOOTH_DCS so far:' + str(np.mean(smooth_dice_scores)))
        
    if output_probability:
        img_probs = nib.Nifti1Image(img_probs, affine)
        return img_probs, output, dsc, smooth_dice_scores,foreground_percent_list
    else:
        img_binary = nib.Nifti1Image(img_binary, affine)
        return img_binary, output, dsc, smooth_dice_scores,foreground_percent_list
    
##########################################################################################################################

def segment(configFile,workingDir):
    '''
    Main segmentation function. Called directly from SEGMENT.py
    
    Args:
        configFile: Path to configuration file.
        workingDir: Path to working directory.
    
    Returns:
        None

    Performs segmentations with a trained model.
    Saves output segmentations in SessionFolder/predictions/
    '''

    #Add scripts folder to path
    workingDir = workingDir+'/biomedical_segmenter'
    scripts_path = configFile.split('configFiles')[0] + 'scripts'
    sys.path.append(scripts_path)
    
    #Import config file
    path = '/'.join(configFile.split('/')[:-1])
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)
    cfg = __import__(configFileName)
    
    epoch = 0
    
    #Move into session folder
    os.chdir(workingDir + '/training_sessions/')
    session = cfg.session
    wd = workingDir + '/training_sessions/' +session
    print('\n CURRENTLY IN SESSION {} \n'.format(session))
    if not os.path.exists(wd):    
        os.mkdir(session)
        os.mkdir(session + '/models')
        os.mkdir(session + '/predictions')
    os.chdir(wd) #Update working directory to session folder
    
    logfile = 'segmentations.log'
    dice_compare = cfg.dice_compare
    cfg.segmentChannels = [workingDir + x for x in cfg.segmentChannels]
    if len(cfg.segmentLabels) > 0:
        cfg.segmentLabels = workingDir + cfg.segmentLabels 
        dice_compare = True
    if len(cfg.TPM_channel) != 0:
        cfg.TPM_channel = workingDir + cfg.TPM_channel
        dice_compare = False

    from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
    my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                            'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                            'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}
    model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )

    full_segm_DICE = []
    full_segm_SMOOTH_DICE = []
    np.set_printoptions(precision=3)

    print("------------------------------------------------------")
    print("                 WHOLE SCAN SEGMENTATION")
    print("------------------------------------------------------")
    dsc = []
    score_smooth = []
    foreground_percent_list = []
    epoch_foreground_percent = []

    with open(cfg.segmentChannels[0]) as vl:
        n_segmentSubjects = len(vl.readlines())
    if cfg.test_subjects > n_segmentSubjects:
        print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
        subjects in test set (" +str(n_segmentSubjects)+ ")")
        cfg.test_subjects = n_segmentSubjects
        print('Using {} number of test subjects'.format(n_segmentSubjects))
    if len(cfg.list_subjects_fullSegmentation) == 0:
      list_subjects_fullSegmentation = range(cfg.test_subjects)
    else:
      list_subjects_fullSegmentation = cfg.list_subjects_fullSegmentation
    for subjectIndex in list_subjects_fullSegmentation: 
        t_segment = time.time()
        
        #TODO: Remove unused parameters from fullSegmentation
        segmentation_img, output, dsc, score_smooth,foreground_percent = fullSegmentation(wd, cfg.penalty_MATRIX, cfg.resolution, cfg.OUTPUT_PATH, cfg.TPM_channel, dice_compare, dsc, score_smooth, 
                                                                                              foreground_percent_list, model, cfg.segmentChannels, cfg.segmentLabels, subjectIndex,
                                                                                              cfg.output_classes, cfg.segmentation_dpatch, cfg.size_test_minibatches,cfg.output_probability, 
                                                                                              cfg.use_coordinates, cfg.percentile_normalization, cfg.model_patch_reduction, cfg.model_crop, 
                                                                                              epoch, using_Unet=True, using_unet_breastMask=cfg.using_unet_breastMask)
 
        if dice_compare: 
            my_logger('--------------- TEST EVALUATION ---------------', logfile)
            my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
            LABEL_CHANNEL = open(cfg.segmentLabels).readlines()[subjectIndex][:-1]
            if 'BENIGN' in LABEL_CHANNEL:
                my_logger('foreground_percent {}'.format(foreground_percent[-1]), logfile)
                my_logger('SMOOTH_DCS N/A',logfile)
                my_logger('DCS N/A',logfile)
            else:
                my_logger('foreground_percent N/A', logfile)
                my_logger('SMOOTH_DCS ' + str(score_smooth[-1]),logfile)  
                my_logger('DCS ' + str(dsc[-1]),logfile)

        if cfg.saveSegmentation and not os.path.exists(output):
            nib.save(segmentation_img, output)
            my_logger('Saved segmentation of subject at: ' + output, logfile)
            
        print('Segmentation of subject took {} s'.format(time.time()-t_segment))
    my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
    full_segm_DICE.append(np.mean(dsc))   
    full_segm_SMOOTH_DICE.append(np.mean(score_smooth))   
    my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
    my_logger('Overall SMOOTH_DCS:   ' + str(full_segm_SMOOTH_DICE[-1]),logfile)
    epoch_foreground_percent.append(np.mean(foreground_percent_list))            
    my_logger('Epoch_foreground_percent {}'.format(epoch_foreground_percent[-1]), logfile)

############################# MODEL TRAINING AND VALIDATION FUNCTIONS ############################################

#batch = return_dict['TRAINING'][0]
#labels = return_dict['TRAINING'][1]
#TPM_patches = return_dict['TRAINING'][3] 
#coords = return_dict['TRAINING'][2]
#size_minibatches = cfg.size_minibatches

def train_validate_model_on_batch(model_name, model,context,batch,labels,coords,TPM_patches,size_minibatches,history,losses,metrics,output_classes,logfile=0, TRAINING_FLAG=True, using_unet_breastMask=False, verbose=False):
    batch_performance = []   
    INPUT_DATA = []
    
    if 'UNet' in model_name:
        if using_unet_breastMask:
            batch = batch[:,:,:,:,0]
            batch = batch.reshape(batch.shape + (1,))
        INPUT_DATA.append(batch)
        if len(coords) > 0:
          #coords = coords.reshape( (coords.shape[0],) + (1,) + coords.shape[1:] )  
          INPUT_DATA.append(coords)
       
    else:    
        # Context
        INPUT_DATA.append(context.reshape(context.shape + (1,)))    
        
        for jj in range(batch.shape[-1]):
          INPUT_DATA.append(batch[:,:,:,:,jj].reshape(batch[:,:,:,:,jj].shape + (1,)))
          
        if len(TPM_patches) > 0:
          INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
    
        if len(coords) > 0:
          #coords = coords.reshape( (coords.shape[0],) + (1,) + coords.shape[1:] )  
          INPUT_DATA.append(coords)

    ######### TRAINING ###########
    if TRAINING_FLAG:
        print('Training..')
        #model.fit(INPUT_DATA, np.float32(labels), verbose = 1, callbacks = [history], batch_size = size_minibatches)
        #Implementing LR scheduler
        model.fit(INPUT_DATA, np.float32(labels), verbose = 1, callbacks = [history], batch_size = size_minibatches)

        if verbose:
          freq = classesInSample(labels, output_classes)
          print("Sampled following number of classes in training MINIBATCH: " + str(freq))
      
        if logfile != 0:
            output_results = zip(['Train cost and metrics     ']*len(history.losses), history.losses, history.metrics)
            for line in output_results:
                my_logger(' '.join(map(str, line)),logfile,print_out=False)
            
    ######### VALIDATION ###########
    else:
        print('Validation..')
        batch_performance.append(model.evaluate(INPUT_DATA, np.float32(labels), verbose=1, batch_size = size_minibatches))
        
    del batch
    del labels
    if TRAINING_FLAG:
        return history.losses, history.metrics
    else:    
        val_performance = np.mean(batch_performance, 0)
        my_logger('Validation cost and accuracy ' + str(val_performance),logfile)                    
        return list(val_performance)   


################################ MAIN TRAINING FUNCTION ###########################################

def train_test_model(configFile, workingDir):
    '''
    Main training function. Called directly from TRAIN_TEST.py
    
    Args:
        configFile: Path to configuration file.
        workingDir: Path to working directory.
    
    Returns:
        None

    Performs training and validation of a model.
    Saves the trained model and the training history in a session specific folder defined through the provided configuration file.
    '''
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.models import load_model  
    from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
    from tensorflow.keras.optimizers import Adam

    #Importing configuration file
    print(configFile)
    path = '/'.join(configFile.split('/')[:-1])
    print(path)
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)

    #Loading in configuration file and determining additional parameters
    cfg = __import__(configFileName)
    if len(cfg.TPM_channel) != 0:
      cfg.TPM_channel = workingDir + cfg.TPM_channel
    cfg.trainChannels = [workingDir + x for x in cfg.trainChannels]
    cfg.trainLabels = workingDir +cfg.trainLabels 
    cfg.testChannels = [workingDir + x for x in cfg.testChannels]
    cfg.testLabels = workingDir + cfg.testLabels
    cfg.validationChannels = [workingDir + x for x in cfg.validationChannels]
    cfg.validationLabels = workingDir +cfg.validationLabels
    
    # Load in the axial pre-trained model for fine tuning
    my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                            'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                            'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

    #Load pre-trained model
    model0 = load_model(cfg.path_to_model, custom_objects = my_custom_objects)
    model0.trainable = False #Disable training of the first model
    
    #Add additional layers to pretrained model using sequential API
    #Functional API introduced segmentation artifacts in initial testing!
    inputs = tf.keras.layers.Input(shape=(None,None,None,3))
    preproc_layer = tf.keras.layers.Dense(3, activation='linear', use_bias=True, kernel_initializer='identity', bias_initializer=tf.keras.initializers.Zeros())
    model = tf.keras.models.Sequential([
        inputs,
        preproc_layer,
        model0
    ])

    #Print model summary and compile for training
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(learning_rate=cfg.learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
    model.summary()    
    
    #Generating and moving working directory into session specific folder
    os.chdir(workingDir + '/training_sessions/')
    session = cfg.model + '_' + cfg.dataset + '_' + configFileName + '_' + time.strftime("%Y-%m-%d_%H%M") 
    wd = workingDir + '/training_sessions/' +session
    if not os.path.exists(wd):    
        os.mkdir(session)
        os.mkdir(session + '/models')
        os.mkdir(session + '/predictions')
    os.chdir(wd)#Updating working directory to session directory

    #Configure CV_FOLD_ARRAYS directory
    CV_FOLDS_ARRAYS_PATH = './CV_FOLDS_ARRAYS/'
    if not os.path.exists(CV_FOLDS_ARRAYS_PATH):
        os.mkdir(CV_FOLDS_ARRAYS_PATH)
    
    #Save model plot, model summary, and any comments into session directory
    copy(workingDir + '/' + configFile, wd)
    logfile = session +'.log'            
    print(model.summary())
    #val_performance = []
    plot_model(model, to_file=wd+'/multiscale_TPM.png', show_shapes=True)
    with open(wd+'/model_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    if len(cfg.comments) > 0:
        f = open('Comments.txt','w')
        f.write(str(cfg.comments))
        f.close()

    #################################################################################################
    #                                                                                               #
    #                       START SESSION                                         #
    #                                                                                               #
    #################################################################################################
    
    #Initialize metrics
    val_performance = []
    full_segm_DICE = []
    full_segm_DICE_train = []
    full_segm_SMOOTH_DICE = []
    full_segm_SMOOTH_DICE_train = []
    epoch_foreground_percent = []
    losses = []
    metrics = []
    EARLY_STOP = False      

    #Logging training parameters TODO: Remove deprecated parameters from start_training_session_logger
    start_training_session_logger(logfile, cfg.threshold_EARLY_STOP, cfg.TPM_channel, cfg.saveSegmentation, cfg.path_to_model, model, \
        cfg.dropout, cfg.trainChannels, cfg.trainLabels, cfg.validationChannels, cfg.validationLabels, \
        cfg.testChannels, cfg.testLabels, cfg.num_iter, cfg.epochs, cfg.n_patches, cfg.n_patches_val, cfg.n_subjects, cfg.samplingMethod_train, \
        cfg.size_minibatches, cfg.n_full_segmentations, cfg.epochs_for_fullSegmentation, cfg.size_test_minibatches)
    
    # Callback history    
    history = LossHistory_multiDice2() 
    saveloc = '/models/best_model.h5' #Location for the model to be saved

    # Training loop
    for epoch in range(0,cfg.epochs):
        t1 = time.time()
        gc.collect()
        my_logger("######################################################",logfile)
        my_logger("                   TRAINING EPOCH " + str(epoch) + "/" + str(cfg.epochs),logfile)
        my_logger("######################################################",logfile)
    
        ####################### FULL HEAD SEGMENTATION ##############################

        #If epoch is in the list of epochs for full segmentation, perform full segmentation          
        if epoch in cfg.epochs_for_fullSegmentation:

            #Log full segmentation
            my_logger("------------------------------------------------------", logfile)
            my_logger("                 FULL HEAD SEGMENTATION", logfile) #TODO: Logger requires specific text for extraction
            my_logger("------------------------------------------------------", logfile)
            
            #Initialize variables
            dice_compare = True
            dsc = []
            dsc_train = []
            smooth_dice_scores = []
            smooth_dice_scores_train = []
            foreground_percent_list = []
            subjectIndex = 0

            #Read number of validation samples
            with open(cfg.validationLabels) as vl:
                n_valSubjects = len(vl.readlines())
            #If number of subjects for testing is larger than the number of validation subjects, set it to the number of validation subjects
            if cfg.test_subjects > n_valSubjects:
                print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
                subjects in test set (" +str(n_valSubjects)+ ")")
                cfg.test_subjects = n_valSubjects
                cfg.n_full_segmentations = n_valSubjects
                print('Using {} number of test subjects'.format(n_valSubjects))

            #If no subjects listed for full segmentation
            if len(cfg.list_subjects_fullSegmentation) == 0:
                list_subjects_fullSegmentation = random.sample(range(cfg.test_subjects ), cfg.n_full_segmentations)
            #If subjects listed for full segmentation, load from config file
            else:
                list_subjects_fullSegmentation = cfg.list_subjects_fullSegmentation

            #for each subject in the list of subjects for full segmentation
            for subjectIndex in list_subjects_fullSegmentation: 
                t_segment = time.time() #Track starting time of each segmentation

                #TODO: Remove deprecated parameters from fullSegmentation
                segmentation_img, output, dsc, smooth_dice_scores,foreground_percent_list = fullSegmentation(wd, cfg.penalty_MATRIX,cfg.resolution, cfg.OUTPUT_PATH, cfg.TPM_channel, 
                                                                                                    dice_compare, dsc, smooth_dice_scores, foreground_percent_list, 
                                                                                                    model, cfg.testChannels, cfg.testLabels, subjectIndex, 
                                                                                                    cfg.output_classes, cfg.segmentation_dpatch, cfg.size_test_minibatches,
                                                                                                    cfg.output_probability, cfg.use_coordinates, cfg.percentile_normalization,
                                                                                                    cfg.model_patch_reduction, cfg.model_crop, epoch, using_Unet=True, 
                                                                                                    using_unet_breastMask=cfg.using_unet_breastMask)

                #Log segmentation results and save output
                LABEL_CHANNEL = open(cfg.testLabels).readlines()[subjectIndex][:-1]
                my_logger('--------------- TEST EVALUATION ---------------', logfile)
                my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
                #If benign, only log foreground percent
                if 'BENIGN' in LABEL_CHANNEL: #Removed because logging values from all samples
                    my_logger('foreground_percent {}'.format(foreground_percent_list[-1]), logfile)
                    my_logger('SMOOTH_DCS N/A',logfile)
                    my_logger('DCS N/A',logfile)
                #If malignant, log DCS
                else:
                    my_logger('foreground_percent N/A', logfile)
                try: 
                    my_logger('SMOOTH_DCS ' + str(smooth_dice_scores[-1]),logfile)
                except:
                    continue
                my_logger('DCS ' + str(dsc[-1]),logfile)

                #Save segmentation
                if cfg.saveSegmentation:
                    nib.save(segmentation_img, output)
                    my_logger('Saved segmentation of subject at: ' + output, logfile)    
                
                
                #Print time between completion and start of segmentation
                print('Segmentation of subject took {} s'.format(time.time()-t_segment))
            
            #If dsc is not None, log full segmentation summary statistics
            if dsc != None:
                my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
                full_segm_DICE.append(np.mean(dsc))
                full_segm_SMOOTH_DICE.append(np.mean(smooth_dice_scores))
                full_segm_DICE_train.append(np.mean(dsc_train))
                full_segm_SMOOTH_DICE_train.append(np.mean(smooth_dice_scores_train))
                my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
                my_logger('Overall SMOOTH_DCS:   ' + str(full_segm_SMOOTH_DICE[-1]),logfile)
                my_logger('Overall_DCS_train:   ' + str(full_segm_DICE_train[-1]),logfile)
                my_logger('Overall_SMOOTH_DCS_train:   ' + str(full_segm_SMOOTH_DICE_train[-1]),logfile)

                epoch_foreground_percent.append(np.mean(foreground_percent_list))            
                my_logger('Epoch_foreground_percent {}'.format(epoch_foreground_percent[-1]), logfile)

            # Function to define if STOP flag goes to True or not, based on difference between last three or two segmentations.
            if len(full_segm_DICE) > 5: # if more than 5 segmentations have been performed
                #if the maximum change between epochs is less than the threshold, set EARLY_STOP to True                        
                if np.max(np.abs(np.diff([full_segm_DICE[-3], full_segm_DICE[-2], full_segm_DICE[-1]] ))) < cfg.threshold_EARLY_STOP:
                    EARLY_STOP = True


            #If only one segmentation has been performed, save initial model
            if len(full_segm_DICE) == 1:
                my_logger('###### SAVING TRAINED MODEL AT : ' + wd + saveloc, logfile)

            #Stop training and log if EARLY_STOP is True
            if EARLY_STOP:
                my_logger('Convergence criterium met. Stopping training.',logfile)
                break           
        #################################################################################################
        #                                                                                               #
        #                               Training and Validation                                         #
        #                                                                                               #
        #################################################################################################
        ####################################### PARALLELIZED TRAINING ITERATIONS ########################################
    
        #For each number of iterations (batches) specified in the config file
        for i in range(0, cfg.num_iter):
            print("###################################################### ")          
            my_logger("                   Batch " + str(i+1) + "/" + str(cfg.num_iter) ,logfile)
            print("------------------------------------------------------ ")          
                        
            ####################### VALIDATION ON BATCHES ############################      
            print('\n###################### VALIDATION ####################')                                      
            with open(cfg.validationLabels) as vl:
                n_valSubjects = len(vl.readlines())
            if cfg.n_subjects_val > n_valSubjects:
                print("Given number of subjects for test set (" + str(cfg.n_subjects_val) +") is larger than the amount of subjects in test set (" +str(n_valSubjects)+ ")")
                cfg.n_subjects_val = n_valSubjects
                print('Using {} number of test subjects'.format(n_valSubjects))
        
            context, patches, target_labels, spatial_coordinates, TPM_patches = sampleTrainData_daemon('VALIDATION',cfg.resolution, cfg.validationChannels, CV_FOLDS_ARRAYS_PATH, cfg.validationLabels, cfg.TPM_channel, cfg.n_patches_val, 
                                                                                cfg.n_subjects_val, cfg.dpatch, cfg.output_classes, cfg.samplingMethod_val, cfg.use_coordinates, 
                                                                                cfg.proportion_malignants_to_sample_val, cfg.percentile_voxel_intensity_sample_benigns, 
                                                                                cfg.data_augmentation, cfg.proportion_to_flip, cfg.percentile_normalization,  cfg.model_patch_reduction, cfg.model_crop,
                                                                                cfg.balanced_sample_subjects)

            val_performance.append(train_validate_model_on_batch(cfg.model, model, context, patches, target_labels, spatial_coordinates, TPM_patches, cfg.size_minibatches_val, history, losses,  metrics, 
                                                                cfg.output_classes, logfile, TRAINING_FLAG=False, using_unet_breastMask=cfg.using_unet_breastMask)) 
            
            # MODEL SELECTION : VALIDATION
            # Save model if current segmentation is better than any previous segmentation
            # First attempt at early stopping on entire validation set results
            if len(val_performance) > 1:
                if np.max([val_performance[i][-1]for i in range(len(val_performance)-1)]) <= val_performance[-1][-1]:
                    my_logger('###### SAVING TRAINED MODEL AT : ' + wd + saveloc, logfile)
                    model.save(wd+saveloc)

            #--------- TRAINING ---------
            print('\n###################### TRAINING ####################')                                                   
            with open(cfg.trainLabels) as vl:
                n_trainSubjects = len(vl.readlines())                
            if cfg.n_subjects > n_trainSubjects:
                print("Given number of subjects for test set (" + str(cfg.n_subjects) +") is larger than the amount of \
                subjects in test set (" +str(n_trainSubjects)+ ")")
                cfg.n_subjects = n_trainSubjects
                print('Using {} number of test subjects'.format(n_trainSubjects))
        
            context, patches, target_labels, spatial_coordinates, TPM_patches = sampleTrainData_daemon('TRAINING', cfg.resolution,cfg.trainChannels, CV_FOLDS_ARRAYS_PATH, cfg.trainLabels,cfg.TPM_channel, 
                                                                        cfg.n_patches, cfg.n_subjects, cfg.dpatch, cfg.output_classes, 
                                                                        cfg.samplingMethod_train, cfg.use_coordinates, cfg.proportion_malignants_to_sample_train,
                                                                        cfg.percentile_voxel_intensity_sample_benigns,  cfg.data_augmentation, cfg.proportion_to_flip, 
                                                                        cfg.percentile_normalization, cfg.model_patch_reduction, cfg.model_crop, cfg.balanced_sample_subjects)
            
            print('patches.shape = {}'.format(patches.shape))
            print('target_labels.shape = {}'.format(target_labels.shape))

            # Train model
            epoch_loss, epoch_metrics = train_validate_model_on_batch(cfg.model, model,context,patches,
                                                                target_labels, spatial_coordinates, TPM_patches,
                                                                cfg.size_minibatches,history,losses,metrics,cfg.output_classes, logfile, using_unet_breastMask=cfg.using_unet_breastMask)  
            try:
                global_loss = np.concatenate([np.load(wd + '/LOSS.npy', allow_pickle=True), epoch_loss])
                global_metrics = np.concatenate([np.load(wd + '/METRICS.npy', allow_pickle=True), epoch_metrics]) 
                np.save(wd + '/LOSS.npy', global_loss)
                np.save(wd + '/METRICS.npy', global_metrics)
            except:
                np.save(wd + '/LOSS.npy', epoch_loss)
                np.save(wd + '/METRICS.npy', epoch_metrics)              
        
        my_logger('Total training this epoch took ' + str(round(time.time()-t1,2)) + ' seconds',logfile)
        
        #Track dense layer fine tuning process, displaying both weights and bias
        weights = model.get_weights()
        my_logger('-----------Trained Model Dense Weights [identity init]-----------',logfile)
        my_logger(weights[0],logfile)
        my_logger('-----------Trained Model Dense Bias [zero init]-----------', logfile)
        my_logger(weights[1], logfile)

        my_logger(f'Current learning rate: {model.optimizer.learning_rate.value()}', logfile)
      #############################################################################################################################
        
    model.save(wd+'/models/modelAtLastEph.h5') #Save final
    return #End of train_test_model