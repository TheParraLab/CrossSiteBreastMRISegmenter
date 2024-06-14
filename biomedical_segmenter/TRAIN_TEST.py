#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:36:27 2018

@author: lukas
"""

"""
This script is used to train and fine-tune a model.
To use, call: python TRAIN_TEST.py </path_to/config.py>

See ./configFiles/configFile_fineTuneKFold.py for example config files.
"""

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    os.chdir('./biomedical_segmenter')
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd() + '/scripts')

    if len(sys.argv) < 2:
        print('Please include a training configuration file:')
        print('>>> python TRAIN_TEST.py </path_to/config.py> \n')
        sys.exit()

    workingDir = os.getcwd()
 
    if not os.path.exists(workingDir):
        os.mkdir(workingDir)
    os.chdir(workingDir)

    from scripts.lib_andy_simple_axl_fineTune import train_test_model
    configFile = sys.argv[1:][0]
    train_test_model(configFile, workingDir)