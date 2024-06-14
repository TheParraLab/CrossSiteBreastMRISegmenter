# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:30:56 2019

@author: hirsch
"""

import keras.backend as K
#------------------------------------------------------------------------------------------
'''
This file contains the loss and accuracy metrics used for training and evaluating the models.

The metrics are defined in the following order:
    - dice_coef
    - Generalised_dice_coef_multilabel2
    - dice_coef_multilabel_bin0
    - dice_coef_multilabel_bin1
'''

def dice_coef(y_true, y_pred):
    """
    This is the basic dice coefficient for measuring overlap between two masks. A perfect overlap returns 1. A total disagreement returns 0.
    
    Args:
        y_true: The true mask
        y_pred: The predicted mask
    Returns:
        The dice coefficient
    """
    
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred):
    """
    This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels
    Assumes that the labels are in the last dimension and two labels are present.
    """
    dice=0
    for index in range(2):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return 2 + dice

def dice_coef_multilabel_bin0(y_true, y_pred):
    "Calculates the dice coefficient for the background class"
    dice = dice_coef(y_true[:,:,:,:,0], K.round(y_pred[:,:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1(y_true, y_pred):
    "Calculates the dice coefficient for the foreground class"
    dice = dice_coef(y_true[:,:,:,:,1], K.round(y_pred[:,:,:,:,1]))
    return dice


