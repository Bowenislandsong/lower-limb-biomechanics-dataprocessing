import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# from algo.lower_limb_data_classes import combine_data
# from algo.util import *
# from algo.extract_data import get_right_gc, step_segmentation

# def get_all_data():
#     base_path = "/media/champagne/lower_limb_dataset/"
#     subject_paths = glob.glob(os.path.join(base_path, "AB*/*_*_*"))
#     subject_ids,dates = [],[]
#     for file_path in subject_paths:
#         subject_id = file_path.split('/')[-2]
#         date = file_path.split('/')[-1]

#         imu_data = combine_data(file_path, signal_sources=['imu', 'gcLeft', 'gcRight'])
#         grf_data = combine_data(file_path, signal_sources=['fp'])

# def process_subject(imu_path, grf_path):
#     print(f"Processing {imu_path}, {grf_path}")
#     imu_data = pd.read_pickle(imu_path)
#     grf_data = pd.read_pickle(grf_path)
#     for session in get_unique_sessions(imu_data, session_title='session', prefix='treadmill'):
#         window_size = 300
#         rgc_data = get_right_gc(imu_data, session)

#         print("Processing session:", session)
#         segmented_imus = step_segmentation(imu_data, session, rgc_data, window_size=window_size)
#         segmented_grfs = step_segmentation(grf_data, session, rgc_data, window_size=window_size, include_columns=['Treadmill_R_v'], dims=['x', 'y', 'z'])

#         yield segmented_imus, segmented_grfs


# def get_all_subjects():
#     # base_path = "./dataset/pandas/"
#     base_path = "/media/champagne/lower_limb_dataset/v2/"
#     imu_paths = glob.glob(os.path.join(base_path, "AB*_IMU.pkl"))
#     grf_paths = [p.replace('IMU','GRF') for p in imu_paths]
#     Parallel(n_jobs=-1)(delayed(process_subject)(imu_path, grf_path) for imu_path, grf_path in zip(imu_paths, grf_paths))

import re

def extract_parts(filename):
    match = re.match(r'(AB\d{2})_.*treadmill_(.*)', filename)
    if match:
        return match.groups()
    print(f"Error extracting parts from {filename}")    
    return None

#package npz files into a single file
def package_npz_files():
    files = glob.glob(os.path.join("./local_results/", "AB*treadmill*.npz"))
    subject_info = pd.read_csv("./dataset/raw/SubjectInfo.csv")
    result = np.array([])
    extra_info = pd.DataFrame(columns=[*subject_info.columns])
    for f in files:
        fname = f.split('/')[-1].replace('.npz','')
        data = np.load(f)
        id,session = extract_parts(fname)
        body_weight = subject_info.loc[subject_info['Subject'] == id]['Weight'].values[0]*2.20462
        data = np.concatenate((data['imu'].transpose(1,0,2,3,4), np.expand_dims(data['grf'], axis=0)/body_weight), axis=0)

        result = np.concatenate((result, data), axis=1) if result.size else data



        for sp in [0.5,1.2,1.55,0.85]: # speeds in m/s
            session_info = pd.DataFrame(subject_info.loc[subject_info['Subject'] == id].to_dict(orient='records'))
            session_info['speed'] = sp+int(session.split('_')[0])*0.05
            session_info['session'] = session
            extra_info = pd.concat([extra_info, session_info], ignore_index=True)
    extra_info = extra_info.rename(columns={extra_info.columns[-1]: 'session'})
    signal_names = ["shank_Accel", "shank_Gyro", "trunk_Accel", "trunk_Gyro", "GRF"]
    data_dimension_description = {'signal_names':signal_names,"session_info":extra_info, "right steps":result.shape[2],'dimensions':['x','y','z'], 'time_steps':np.arange(result.shape[-1])*1/200}
    print(result.shape, extra_info.shape)
    print(extra_info)

    np.savez(f"./local_results/locomotion_data.npz", data=result, **data_dimension_description)

if __name__ == "__main__":
    # get_all_data()
    # get_all_subjects()
    package_npz_files()