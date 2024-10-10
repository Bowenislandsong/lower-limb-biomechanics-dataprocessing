# Lower Limb Biomechanics Dataset Preparation

## Dataset Source:
Camargo, Jonathan, et al. "A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions." Journal of Biomechanics 119 (2021): 110320.
## Dataset Description:
This dataset contains lower limb biomechanics data collected from multiple participants performing various ambulation tasks, including stairs, ramps, and level-ground walking, as well as transitions between these conditions.


## Preprecessing

This repo is dedicated to preprocess this dataset. We are to use MATLAB to convert all .mat files to csv files. See [matlab](./matlab) folder for it. It requires to use MATLAB to run the code. 

THen, we will use [data class](./algo/lower_limb_data_classes.py) to create .pkl files. Specifically for IMU and GRF. We can also generate for other signals, but I am only interested in those 2 signals. 

We will be able to plot them.
 We then use  [extract data](./algo/extract_data.py) to make them into .npz files that contains IMU and GRF signals. THey are both orgnized to be per collection sesssion. They contain.  1 person, 4 speeds, 4 signals, 15 steps, 3 dimentions, time window. We can then use these files to do further analysis. 

## Preprocessing Lower Limb Dataset
This repository focuses on preprocessing a dataset containing lower limb motion data.

Workflow:

1. Convert .mat files to csv: The matlab folder contains scripts that utilize MATLAB to convert all .mat files in the dataset to comma-separated values (csv) format. This makes the data more accessible for further processing.

2. Create .pkl files for specific signals:  The algo/lower_limb_data_classes.py script (written in Python) generates pickle (.pkl) files for Inertial Measurement Unit (IMU) and Ground Reaction Force (GRF) data. You can modify this script to create .pkl files for other signals if needed.

3. Extract data and generate .npz files: The algo/extract_data.py script extracts data from the csv files and organizes it into NumPy archive (.npz) files. Each .npz file is structured per collection session and contains data for:

- 1 person
- 4 walking speeds
- 4/1 signals (shank accleration, trunk gyro, etc)
- 15 steps per speed
- 3 dimensions (likely X, Y, and Z)
Time window for each data point (currently at 300 time points.)