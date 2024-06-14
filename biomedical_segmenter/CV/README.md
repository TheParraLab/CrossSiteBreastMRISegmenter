# Data Linking
The files within the `CV` directory contain lists to all necessary data files.  Data is seperated by channel, with the order of T1post, Slope1, and Slope2.  The files are referenced by the configuration files to link the data to the session.

## Format
The data files are stored in a .txt format.  Each line contains the filepath to a single data file.  The ordering of samples must be identical across files.

## get_paths.py
This script will generate the data files for the public data.  The script will search for all data files within the `PublicData` directory and write the filepaths to the appropriate .txt files.  The script will output the following files:
- `PublicData_allT1post.txt` - T1 post-contrast images
- `PublicData_allSlope1.txt` - Slope 1 images
- `PublicData_allSlope2.txt` - Slope 2 images
- `PublicData_alltrainlabels.txt` - Training labels