import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
import itertools
import numpy as np
from scipy.signal import find_peaks
import os
import glob

def get_data_for_session(imu_data, grf_data, session, locomotion='treadmill', include_imu_columns=['shank_', 'trunk_','_HeelStrike'], include_grf_columns=['Treadmill_L_v','Treadmill_R_v']):
    # print sesssions, if the session is not in the data.
    if session not in imu_data['session'].unique():
        print(f"Session {session} not found in the IMU data. Available sessions: {imu_data['session'].unique()}")
    if session not in grf_data['session'].unique():
        print(f"Session {session} not found in the GRF data. Available sessions: {grf_data['session'].unique()}")

    imu_data_filtered = imu_data[
        (imu_data['session'] == session) | 
        ('gcLeft_session' in imu_data.columns and imu_data['gcLeft_session'] == session) | 
        ('gcRight_session' in imu_data.columns and imu_data['gcRight_session'] == session)
    ]
    imu_data_filtered = imu_data_filtered[imu_data_filtered['locomotion_mode'] == locomotion]
    grf_data_filtered = grf_data[(grf_data['session'] == session)]
    grf_data_filtered = grf_data_filtered[grf_data_filtered['locomotion_mode'] == locomotion]

    # combine the data
    data = pd.merge(imu_data_filtered, grf_data_filtered, left_index=True, right_index=True, suffixes=('_imu', '_grf'))
    # Filter the columns to include only the specified columns
    imu_columns_to_include = [col for col in data.columns if any([pattern in col for pattern in include_imu_columns])]
    grf_columns_to_include = [col for col in data.columns if any([pattern in col for pattern in include_grf_columns])]
    data = data[imu_columns_to_include + grf_columns_to_include]

    return data

# break an array into n segments, take a fixed number of elements from each segment from the middle.
def segment_data(data, n_segments, n_elements):
    segment_size = len(data) // n_segments
    segmented_data = []
    for i in range(n_segments):
        segment = data[i*segment_size:(i+1)*segment_size]
        middle_index = len(segment) // 2
        start_index = max(0, middle_index - n_elements // 2)
        end_index = start_index + n_elements
        segmented_data.append(segment[start_index:end_index])
    return np.asarray(segmented_data)


def step_segmentation(data, window_size=220,
                      n_speeds=4, n_steps=10,
                      indicator='gcRight_HeelStrike', 
                      include_columns=['shank_Accel_','trunk_Accel_','shank_Gyro_','trunk_Gyro_'],
                      dims=['X','Y','Z'], post_padding=False):
    # since the dataset only has right leg instrumented, we will only use the right leg data
    segmented_data = np.zeros((n_speeds, len(include_columns), n_steps, 3, window_size))
    
    # Find heel strike events
    heel_strikes,_ = find_peaks(data[indicator])

    heel_strikes = segment_data(heel_strikes, n_speeds, n_steps)
    # include 5 time points before the heel strike
    heel_strikes = heel_strikes - 1

    # diff = np.diff(heel_strikes, axis=1)
    # mean_diff = np.mean(diff,axis=1).astype(int) - 2

    for speed_idx in range(n_speeds):
        for col_idx, col in enumerate(include_columns):
            for step_idx in range(n_steps):
                for dim_idx, dim in enumerate(dims):
                    idx = heel_strikes[speed_idx][step_idx]
                    ts = data[f"{col}{dim}"].index[idx]
                    closest_idx = np.argmin(np.abs(data[f"{col}{dim}"].dropna().index.to_numpy()) - ts)
                    # Find the closest index to ts
                    # closest_idx = data[f"{col}{dim}"].dropna().index.get_loc(ts)
                    # if post_padding:
                    #     segment = data[f"{col}{dim}"].dropna().iloc[closest_idx:closest_idx + min(window_size,mean_diff[speed_idx])].tolist()
                    #     if len(segment) < window_size:
                    #         segment += [0] * (window_size - len(segment))
                    # else:
                    segment = data[f"{col}{dim}"].dropna().iloc[closest_idx:closest_idx + window_size].tolist()
                    segmented_data[speed_idx,col_idx,step_idx,dim_idx] = segment
    # 1 person, 4 speeds, 4 signals, 15 steps, 3 dimentions, time window
    return segmented_data.squeeze()

import matplotlib.pyplot as plt

def generate_plots_for_segmented_data(segmented_imus, segmented_grfs, window_size, titles, save_path):
    t_ax = np.arange(window_size) * 1 / 200
    fig, axs = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows for speeds, 5 columns (4 for IMU, 1 for GRF)

    for i in range(4):
        for j in range(4):
            for k in range(3):
                axs[i, j].plot(t_ax, segmented_imus[i, j, k, :].T)
                axs[i, j].set_title(f'{titles[j]} at speed {i+1}')
                axs[i, j].set_xlabel('Time')
                axs[i, j].set_ylabel('Signal')

        for k in range(3):
            axs[i, 4].plot(t_ax, segmented_grfs[i, k, :].T)
            axs[i, 4].set_title(f'{titles[4]} at speed {i+1}')
            axs[i, 4].set_xlabel('Time')
            axs[i, 4].set_ylabel('Signal')

    plt.tight_layout()
    plt.savefig(save_path)

def get_all_subjects():
    # base_path = "./dataset/pandas/"
    base_path = "/media/champagne/lower_limb_dataset/v2/"
    imu_paths = glob.glob(os.path.join(base_path, "AB*_IMU.pkl"))
    grf_paths = [p.replace('IMU','GRF') for p in imu_paths]

    for imu_path, grf_path in zip(imu_paths, grf_paths):
        print(f"Processing {imu_path}, {grf_path}")
        imu_data = pd.read_pickle(imu_path)
        grf_data = pd.read_pickle(grf_path)
        for session in imu_data['session'].dropna()[imu_data['session'].dropna().str.startswith("treadmill")].unique():
            print(f"Processing session {session}")
            data = get_data_for_session(imu_data, grf_data, session)
            window_size = 400
            segmented_imus = step_segmentation(data, window_size=window_size)
            segmented_grfs = step_segmentation(data, window_size=window_size, indicator='gcRight_HeelStrike', include_columns=['Treadmill_R_v'], dims=['x','y','z'])
            titles = ['Shank Accel', 'Shank Gyro', 'Trunk Accel', 'Trunk Gyro', 'GRF']
            save_path = f"./local_results/{imu_path.split('/')[-1].replace('_IMU.pkl','')}{session}.png"
            generate_plots_for_segmented_data(segmented_imus, segmented_grfs, window_size, titles, save_path)
            # break
        # break


if __name__ == '__main__':
    # imu_path = '/media/champagne/lower_limb_dataset/v2/AB06_10_09_18_IMU.pkl'
    # grf_path = imu_path.replace('IMU','GRF')
    # imu_data = pd.read_pickle(imu_path)
    # grf_data = pd.read_pickle(grf_path)

    # session = 'treadmill_01_01'


    # data = get_data_for_session(imu_data, grf_data, session)


    # window_size = 400
    # segmented_imus = step_segmentation(data, window_size=window_size)
    # segmented_grfs = step_segmentation(data, window_size=window_size, indicator='gcRight_HeelStrike', include_columns=['Treadmill_R_v'], dims=['x','y','z'])

    # titles = ['Shank Accel', 'Shank Gyro', 'Trunk Accel', 'Trunk Gyro', 'GRF']
    # generate_plots_for_segmented_data(segmented_imus, segmented_grfs, window_size, titles, f"./local_results/{imu_path.split('/')[-1].replace('_IMU.pkl','')}{session}.png")

    get_all_subjects()


