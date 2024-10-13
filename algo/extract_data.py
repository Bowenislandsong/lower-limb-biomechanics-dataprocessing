import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import find_peaks
import os
import glob
from .util import *
from joblib import Parallel, delayed

def get_data_for_session(imu_data, grf_data, session, locomotion='treadmill', include_imu_columns=['shank_', 'trunk_','_HeelStrike'], include_grf_columns=['Treadmill_L_v','Treadmill_R_v']):
    # print sesssions, if the session is not in the data.
    assert 'session' in imu_data.columns, "session column not found in the IMU data"
    assert 'session' in grf_data.columns, "session column not found in the GRF data"
    assert 'locomotion_mode' in imu_data.columns, "locomotion_mode column not found in the IMU data"
    assert 'locomotion_mode' in grf_data.columns, "locomotion_mode column not found in the GRF data"
    assert 'gcLeft_session' in imu_data.columns, "gcLeft_session column not found in the IMU data"
    assert 'gcRight_session' in imu_data.columns, "gcRight_session column not found in the IMU data"
    if session not in imu_data['session'].unique():
        print(f"Session {session} not found in the IMU data. Available sessions: {imu_data['session'].unique()}")
    if session not in grf_data['session'].unique():
        print(f"Session {session} not found in the GRF data. Available sessions: {grf_data['session'].unique()}")
    if session not in imu_data['gcLeft_session'].unique():
        print(f"Session {session} not found in the IMU data. Available sessions: {imu_data['gcLeft_session'].unique()}")
    if session not in imu_data['gcRight_session'].unique():
        print(f"Session {session} not found in the IMU data. Available sessions: {imu_data['gcRight_session'].unique()}")

    imu_data_filtered = imu_data[
        (imu_data['session'] == session) | 
        (imu_data['gcLeft_session'] == session) | 
        (imu_data['gcRight_session'] == session)
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

def step_segmentation(data,session_name,indicator_data,window_size=220,
                      n_speeds=4, n_steps=10,
                      include_columns=['shank_Accel_','trunk_Accel_','shank_Gyro_','trunk_Gyro_'],
                      dims=['X','Y','Z']):
    # since the dataset only has right leg instrumented, we will only use the right leg data
    segmented_data = np.zeros((n_speeds, len(include_columns), n_steps, 3, window_size))
    
    # Find heel strike events
    heel_strikes,_ = find_peaks(indicator_data)
    heel_strikes = segment_data(heel_strikes, n_speeds, n_steps)
    if heel_strikes.shape != (n_speeds, n_steps):
        print(f"Error: heel_strikes shape is {heel_strikes.shape}, expected {(n_speeds, n_steps)}")
        return None
    strike_durations = np.diff(heel_strikes, axis=1).mean(axis=1).astype(int)

    for speed_idx in range(n_speeds):
        for col_idx, col in enumerate(include_columns):
            for step_idx in range(n_steps):
                for dim_idx, dim in enumerate(dims):
                    idx = heel_strikes[speed_idx][step_idx]                                  # Get the index of the current heel strike split into speeds and steps
                    ts = indicator_data.index[idx]                                           # Get the timestamp of the heel strike
                    te = indicator_data.index[idx+strike_durations[speed_idx]]               # Get the timestamp of the end of the stride
                    col_data = get_signal(data, session_name, f"{col}{dim}")                 # Get the signal data for the specified column and dimension
                    closest_start = np.argmin(np.abs(col_data.index.to_numpy() - ts)) - 1    # Find the closest start index to the heel strike
                    closest_end = np.argmin(np.abs(col_data.index.to_numpy() - te)) + 1      # Find the closest end index to the end of the stride
                    segment = col_data.values[closest_start:closest_start + window_size]     # Extract the segment of data for the window size
                    if closest_end - closest_start < len(segment):
                        segment[closest_end - closest_start:] = 0                            # Zero-pad the segment if it is shorter than the window size
                        segment = np.roll(segment, (len(segment)-closest_end + closest_start)//2)                            # Shift the segment to the right

                    segmented_data[speed_idx, col_idx, step_idx, dim_idx] = segment          # Store the segment in the segmented data array
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

def process_subject(imu_path, grf_path):
    print(f"Processing {imu_path}, {grf_path}")
    imu_data = pd.read_pickle(imu_path)
    grf_data = pd.read_pickle(grf_path)
    for session in get_unique_sessions(imu_data, session_title='session', prefix='Treadmill').tolist() + get_unique_sessions(imu_data, session_title='session', prefix='treadmill').tolist():
        window_size = 300
        rgc_data = get_right_gc(imu_data, session)

        print("Processing session:", session)
        segmented_imus = step_segmentation(imu_data, session, rgc_data, window_size=window_size)
        segmented_grfs = step_segmentation(grf_data, session, rgc_data, window_size=window_size, include_columns=['Treadmill_R_v'], dims=['x', 'y', 'z'])
        if segmented_imus is None or segmented_grfs is None:
            print(f"Error processing session {session} from {imu_path}")
            continue
        titles = ['Shank Accel', 'Shank Gyro', 'Trunk Accel', 'Trunk Gyro', 'GRF']
        save_path = f"./local_results/{imu_path.split('/')[-1].replace('_IMU.pkl', '')}{session.lower()}.png"
        generate_plots_for_segmented_data(segmented_imus, segmented_grfs, window_size, titles, save_path)
        # save session data
        save_path = f"./local_results/{imu_path.split('/')[-1].replace('_IMU.pkl', '')}{session.lower()}.npz"
        np.savez(save_path, imu=segmented_imus, grf=segmented_grfs)

def get_all_subjects():
    # base_path = "./dataset/pandas/"
    base_path = "/media/champagne/lower_limb_dataset/"
    imu_paths = glob.glob(os.path.join(base_path, "AB*_IMU.pkl"))
    grf_paths = [p.replace('IMU','GRF') for p in imu_paths]
    Parallel(n_jobs=-1)(delayed(process_subject)(imu_path, grf_path) for imu_path, grf_path in zip(imu_paths, grf_paths))

def get_sample_subject():
    imu_path = "./dataset/pandas/AB25_01_20_2019_IMU.pkl"
    grf_path = "./dataset/pandas/AB25_01_20_2019_GRF.pkl"
    imu_data = pd.read_pickle(imu_path)
    grf_data = pd.read_pickle(grf_path)
    print(imu_data['session'].unique())
    for session in get_unique_sessions(imu_data, session_title='session', prefix='treadmill'):
        rgc_data = get_right_gc(imu_data, session)
        window_size = 300
        print("Processing session:", session, rgc_data.shape)
        segmented_imus = step_segmentation(imu_data, session, rgc_data, window_size=window_size)
        segmented_grfs = step_segmentation(grf_data, session, rgc_data, window_size=window_size, include_columns=['Treadmill_R_v'], dims=['x', 'y', 'z'])
        if segmented_imus is None or segmented_grfs is None:
            print(f"Error processing session {session} from {imu_path}")
            continue
        titles = ['Shank Accel', 'Shank Gyro', 'Trunk Accel', 'Trunk Gyro', 'GRF']
        save_path = f"./{imu_path.split('/')[-1].replace('_IMU.pkl', '')}{session}.png"
        generate_plots_for_segmented_data(segmented_imus, segmented_grfs, window_size, titles, save_path)
        # save session data
        save_path = f"./{imu_path.split('/')[-1].replace('_IMU.pkl', '')}{session}.npz"
        np.savez(save_path, imu=segmented_imus, grf=segmented_grfs)

if __name__ == '__main__':
    get_all_subjects()
    # get_sample_subject()


