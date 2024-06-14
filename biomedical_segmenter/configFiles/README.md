# Configuration Files
These files provide configuration parameters for the training and testing of the biomedical segmenter. The parameters are stored as variables within .py files.  Training configurations are within the base directory, while segmentation configurations are within the 'segmentation' sub-directory.

## Linking Data
Data files are linked to the configuration files via lists of filepaths included in the following variables:
- `trainChannels` - A list of filepaths to the training input data.  These files are generally stored in the 'CV' directory.  They are required to be in the order of T1post, Slope1, and Slope 2
- `trainLabels` - Filepath to the training labels.  This file is generally stored in the 'CV' directory.
- `testChannels` - A list of filepaths to the testing input data.  These files are generally stored in the 'CV' directory.  They are required to be in the order of T1post, Slope1, and Slope 2
- `testLabels` - Filepath to the testing labels.  This file is generally stored in the 'CV' directory.
- `segmentChannels` - A list of filepaths to the input data for segmentation.  These files are generally stored in the 'Data' directory.  They are required to be in the order of T1post, Slope1, and Slope 2

An example configuration file to perform segmentations on the public data is provided at `./segmentation/Segmentation_Example.py`. 

Output segmentations will be saved in the `./biomedical_segmenter/training_sessions/tuned_model/predictions/` directory.
