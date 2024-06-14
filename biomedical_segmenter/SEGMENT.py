#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:36:27 2018

@author: lukas
"""

"""
This script is used to segment a given image using a trained model.
To use, call: python SEGMENT.py </path_to/segmentation_config.py>

See ./configFiles/segmentation/Segmentation_Fold*of*.py for example segmentation config files.
"""

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd() + '/scripts')
     
    if len(sys.argv) < 2:
        print('Please include a model configuration file:')
        print('>>> python SEGMENT.py </path_to/segmentation_config.py> \n')
        sys.exit()

    workingDir = os.getcwd()

    from scripts.lib_andy_simple_axl_fineTune import segment
    configFile = sys.argv[1:][0]
    segment(configFile, workingDir)