# Axial Breast Cancer MRI Segmenter
This is the top-level read me for the crosssite axial breast cancer MRI segmenter.

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Usage](#usage)
  - [Data](#data)
    - [Public Data Source](#public-data-source)
  - [Publication (Under Review)](#publication-under-review)


## Overview
This repository encompasses a cross-site training and evaluation of a previously reported breast cancer MRI segmenter.  The segmenter was initially retrained on a large quantity of private axial data from Memorial Sloan Kettering Cancer Center (MSKCC).  Model performance was then analyzed on additional data from MSKCC, as well as John Hopkins University (JHU) and public data from Duke University. 

## Usage
The repository is structured as follows:
```
└── /biomedical_segmenter
    ├── README.md
    ├── SEGMENT.py
    ├── TRAIN_TEST.py
    ├── configFiles/
    │   ├── README.md
    │   ├── configFile_funeTuneKFold_XofY.py
    │   └── segmentation/
    │       ├── README.md
    │       ├── Segmentation_Example.py
    │       └── Segmentation_FoldXofY.py
    ├── CV/
    │   ├── README.md
    │   ├── PublicData_allSlope1.txt
    │   ├── PublicData_allSlope2.txt
    │   ├── PublicData_allT1post.txt
    │   ├── PublicData_alltrainlabels.txt
    │   └── get_paths.py
    ├── Data/
    │   ├── README.md
    │   ├── LABEL_BENIGN.nii.gz
    │   └── PublicData/
    │       ├── Data/
    │       │   ├── Breast_MRI_001/
    │       │   │   ├── T1_axial_02.nii.gz
    │       │   │   ├── T1_axial_slope1.nii.gz
    │       │   │   └── T1_axial_slope2.nii.gz
    |       |   └── Breast_MRI_070/
    │       │       ├── T1_axial_02.nii.gz
    │       │       ├── T1_axial_slope1.nii.gz
    │       │       └── T1_axial_slope2.nii.gz
    │       └── Segmentation/
    │           ├── Breast_MRI_001_T1subtract.nii.gz
    |           └── Breast_MRI_070_T1subtract.nii.gz
    ├── scripts/
    │   ├── lib_andy_simple_axl_fineTune.py
    │   └── MultiPriors_Models_Collection.py
    └── training_sessions/
        ├── tuned_model/
        │   └── models/
        │       └── tuned_model.h5
        └── initial_model/
            └── models/
                └── initial_model.h5
```

To perform segmentations on the public data, run the following command in the base directory:
```
> python ./biomedical_segmenter/SEGMENT.py ./biomedical_segmenter/configFiles/segmentation/Segmentation_Example.json
```
This will output segmentations in the `./biomedical_segmenter/training_sessions/tuned_model/predictions/` directory.

To run the cross-site training and evaluation, run the following command in the base directory:
```
> python ./biomedical_segmenter/TRAIN_TEST.py ./biomedical_segmenter/configFiles/configFile_fineTuneKFold_XofY.py
```

## Data
Models included in this repository have been trained on private data from MSK, with additional fine-tuning completed with data from JHU. All data provided within the repository is part of a public dataset obtained from Duke University.  The data has been preprocessed and is ready for use in the model.

## Publication (Under Review)
Huang, Y., Leotta, N. J., Hirsch, L., Lo Gullo, R., Hughes, M., Reiner, J., Saphier, N. B., Myers, K., Panigrahi, B., Ambinder, E., Di Carlo, P., Grimm, L. J., Lowell, D., Yoon, S., Ghate, S. V., Parra, L. C., & Sutton, E. J. (2024). Radiologist-level performance across multiple clinical sites of a deep network to segment breast cancers on MRI. **Under review**, The Journal of Imaging Informatics in Medicine.


  
