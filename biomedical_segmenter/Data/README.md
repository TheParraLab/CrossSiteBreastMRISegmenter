# Data
This directory contains the data files used for training and segmentation. Containing data in this directory is not a necessity, as the paths listed in the `CV` directory can be used to link data from other locations.

`LABEL_BENIGN.nii.gz` is a label file used for training.  This file is a binary mask that labels benign regions as 1 and all other regions as 0.  This file is used to train the model to segment benign regions, as it is a completely empty segmentation.

The included example is public data obtained from Duke University.  The data has been preprocessed and is ready for use in the model.  One example is stored in the `./PublicData/Breast_MRI_001/` directory.  The data is stored in a .nii.gz format.  The data is stored in the following files:
    - `T1_axial_02.nii.gz` - T1 post-contrast image
    - `T1_axial_slope1.nii.gz` - Slope 1 image
    - `T1_axial_slope2.nii.gz` - Slope 2 image

The segmentation for this public sample is included in `./PublicData/Segmentation` as `Breast_MRI_001_T1subtract.nii.gz`

### Public Data Source

Saha, A., Harowicz, M.R., Grimm, L.J., Kim, C.E., Ghate, S.V., Walsh, R. and Mazurowski, M.A., 2018. A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features. British journal of cancer, 119(4), pp.508-516. 

[Data Link](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
